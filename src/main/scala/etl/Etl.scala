package etl

import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import scala.collection.immutable.HashMap
import scala.annotation.tailrec
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
        case null => "other"
        case x if x.toLowerCase().contains("android") => "android"
        case x if x.toLowerCase().contains("ios") => "ios"
        case x if x.toLowerCase().contains("windows") => "windows"
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

    val df8 = df7
      .withColumn("label", udf((elem: Boolean) => if (elem) 1 else 0).apply(col("label")))

    interestIndex
      .foreach((elem) => {
        df8.withColumn("interest_"+elem._1, lit(0))
      })

    val df9 = interestIndex
      .foldLeft(df8)((df, interest) => {
        df.withColumn(interest._1, udf((array: mutable.WrappedArray[Int]) => {
          if (array.contains(interest._2)) 1 else 0
        }).apply(col("interestsIndex")))
      })
      .drop("interests")
      .drop("interestsAsNames")
      .drop("interestsIndex")

    val df10 = df9
      .columns
      .foldLeft(df9)((df, col) => {
        if(col.startsWith("interest_")) df
        else {
          new StringIndexer()
            .setInputCol(col)
            .setOutputCol(col + "Index")
            .setHandleInvalid("keep")
            .fit(df)
            .transform(df)
            .drop(col)
          }
      })

    val columns: Array[String] = df10
      .columns

    val finalDFAssembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val finalDF = finalDFAssembler.transform(df10)

    finalDF.show()

    return finalDF
  }
}