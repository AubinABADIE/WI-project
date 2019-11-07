import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.log4j.{Level, Logger}
import scala.collection.immutable.HashMap
import org.apache.spark.sql.functions._
import scala.collection.mutable
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.StringIndexer

object Main extends App {

  override def main(args: Array[String]) {
    
    //Ignores logs that aren't errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    //Creates spark session
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Cortex")
      .getOrCreate

    import spark.implicits._ // << add this after defining spark

    val df = spark
      .read
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .json("data/data-students.json")
      .drop("exchange")
      .drop("bidfloor")
      .drop("timestamp")
      .drop("impid")

    val interests = spark
      .read
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .csv("data/interests.csv")

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
     df6.show()

    spark.close()
  }
}
