import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._


import NaiveBayesCleaning._
import NaiveBayesPredictModel._
import IntererestsCleaning._

object Main extends App {

  override def main(args: Array[String]) {
    
    //Ignores logs that aren't errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("Cortex").setMaster("local")
    val sc = new SparkContext(conf)

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
      .drop("size")


    println("Starting to generate interests")
    val interestsStartTime = System.nanoTime()

    val dfInterests = cleanInterests(spark, df)

    val elapsedTimeInterests = (System.nanoTime() - interestsStartTime) / 1e9

    println("Interests DONE, elapsed time: "+elapsedTimeInterests)

    val df1 = dfInterests.drop(col("id"))

    println("Starting to generate double values for rows")
    val cleaningStartTime = System.nanoTime()

    val df2 = clean(spark, df1)

    val elapsedTimeCleaning = (System.nanoTime() - cleaningStartTime) / 1e9
    println("Cleaning DONE, elapsed time: "+ elapsedTimeCleaning)

    val df3 = df2.drop(col("size")).drop(col("interests"))

    df3.show()

    df3.write.format("csv").save("data/values.csv")
    //NaiveBayerPredictModel(spark, df8, sc)
//

    spark.stop()
    //spark.close()

  }
}
