package etl

import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import scala.collection.immutable.HashMap
import scala.collection.mutable

object Etl {

    def apply(df: DataFrame, interests: DataFrame): DataFrame = {

        val spark = SparkSession.builder().getOrCreate()
        import spark.implicits._

        val interestsCodes = interests
            .as[(String, String)]
            .collect
            .toMap

        val codeToInterestUDF = udf((interestsArray: mutable.WrappedArray[String]) => {
        if (interestsArray == null) mutable.WrappedArray.empty[String]
        else {
            interestsArray.map((code: String) => {
            if (!code.startsWith("IAB")) code.toLowerCase //Not an interest code
            else interestsCodes(code).toLowerCase
            })
        }
        }).apply(col("interests"))

        //OS names cleaning
        val df2 = df.withColumn("os", udf((elem: String) => {
        elem match {
            case x if x.toLowerCase.contains("android") => "android"
            case x if x.toLowerCase.contains("ios") => "ios"
            case x if x.toLowerCase.contains("windows") => "windows"
            case _ => "other"
        }
        }).apply(col("os")))

        //Converting the interests column to an array of interests instead of interests separated by ","
        val df3 = df2
            .withColumn("interests", split($"interests", ",").cast("array<String>"))

        //Converting interest codes to their names and sorting them to avoid redundant data
        val df4 = df3
            .withColumn("interestsAsNames", array_distinct(codeToInterestUDF))

        val mediasMap = df4
            .select("media")
            .distinct()
            .collect()
            .map(element => element.get(0))
            .zip(Stream from 1)
            .map(mediaTuple => mediaTuple._1 -> mediaTuple._2)
            .toMap

        val df5 = df4
            .withColumn("media", udf((elem: String) => mediasMap(elem)).apply(col("media")))

        val interestIndex = df5
            .select("interestsAsNames")
            .distinct()
            .collect()
            .flatMap(element => element.get(0).asInstanceOf[mutable.WrappedArray[String]])
            .distinct
            .zip(Stream from 1)
            .map(interests => interests._1 -> interests._2)
            .toMap

        val df6 = df5
            .withColumn("interestsIndex", udf((elem: mutable.WrappedArray[String]) =>
                elem.map(e => interestIndex(e))).apply(col("interestsAsNames")))

        val df7 = df6
            .withColumn("located", udf((elem: String) => {
                elem match {
                case null => "not located"
                case _ => "located"
                }
            }).apply(col("city")))
        
        return df7
    }
}