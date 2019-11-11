import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import NaiveBayesCleaning._
import NaiveBayesPredictModel._
import IntererestsCleaning._

import scala.collection.mutable

object Main extends App {

  override def main(args: Array[String]) {

    val start = System.nanoTime()
    var jsonFile = "data-students.json"
    var model = "naive"

    if(args.isEmpty) println("Usage: ./jarFile jsonFile [option]\nOptions: \n\t* naive\n\t* lr")
    if(args.length == 2){
      jsonFile = args(0)
      model = args(1)
    }else if (args.length == 1){
      jsonFile = args(0)
    }

    println(s"Starting with $jsonFile file using the $model model")

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
      .json("data/"+jsonFile)

    println("Starting to generate interests")
    var interestsStartTime = System.nanoTime()

    val dfInterests = cleanInterests(spark, df)

    val elapsedTimeInterests = (System.nanoTime() - interestsStartTime) / 1e9

    println("Interests DONE, elapsed time: "+elapsedTimeInterests)

    val df2 = dfInterests.drop(col("id"))

    println("Starting to generate double values for rows")
    val cleaningStartTime = System.nanoTime()

    val df3 = clean(spark, df2)

    val elapsedTimeCleaning = (System.nanoTime() - cleaningStartTime) / 1e9
    println("Cleaning DONE, elapsed time: "+ elapsedTimeCleaning)

    /**
      * After ANOVA drop the following columns
      * os, user, i1, i2, i3, i5, i10, i15, i18, i20, i21, i22, i23
      */
    val df4 = df3.drop("os", "user", "i5", "i10", "i15", "i18", "i20", "i21", "i22", "i23")

//    println("Start to export CSV")
//    interestsStartTime = System.nanoTime()
//
//    df4.coalesce(1) //So just a single part- file will be created
//      .write.mode(SaveMode.Overwrite)
//      .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
//      .option("header","true") //Write the header
//      .csv("data/exportCSV")
//
//    println("Export DONE, elapsed time: "+ ((System.nanoTime() - interestsStartTime) / 1e9))


    println("Start to create model Naive Bayes")
    interestsStartTime = System.nanoTime()
    NaiveBayerPredictModel(spark, df4, sc)

    println("Model created, elapsed time: "+ ((System.nanoTime() - interestsStartTime) / 1e9))

    spark.stop()
    //spark.close()

  }
}
