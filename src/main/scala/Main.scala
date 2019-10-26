import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.log4j.{Level, Logger}
import scala.collection.immutable.HashMap
import org.apache.spark.sql.functions._
import scala.collection.mutable

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

     val osNames = HashMap(
        "android" -> "android",
        "Android" -> "android",
        "ios" -> "ios",
        "iOS" -> "ios",
        "WindowsMobile" -> "windows",
        "Windows" -> "windows",
        "windows" -> "windows",
        "WindowsPhone" -> "windows",
        "Windows Phone OS" -> "windows",
        "Windows Mobile OS" -> "windows",
        "other" -> "other",
        "Unknown" -> "other",
        "Rim" -> "other",
        "WebOS" -> "other",
        "Symbian" -> "other",
        "Bada" -> "other",
        "blackberry" -> "other")

    val df2 = df.withColumn("os", udf((elem: String) => {
      if (elem == null) "other"
      else osNames(elem) 
    }).apply(col("os")))

    df2.select("os").distinct().show()
    df2.show(10)
    spark.close()
  }
}