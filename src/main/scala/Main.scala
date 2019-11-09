import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import NaiveBayesCleaning._
import NaiveBayesPredictModel._
import IntererestsCleaning._
import LoadCleanedData.{result, sc, spark}

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
      //.drop("exchange")
      .drop("bidfloor")
      .drop("timestamp")
      .drop("impid")
      .drop("size")


    //Stratified sampling
//    val trueValue = df.filter("label = true")
//    val falseValue = df.filter("label = false").sample(0.030)
//
//    val df1 = trueValue.union(falseValue)

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

    val df4 = df3.drop(col("interests"))

    println("Start to export CSV")
    interestsStartTime = System.nanoTime()

    df4.coalesce(1) //So just a single part- file will be created
      .write.mode(SaveMode.Overwrite)
      .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
      .option("header","true") //Write the header
      .csv("data/exportComplete.csv")

    println("Export DONE, elapsed time: "+ ((System.nanoTime() - interestsStartTime) / 1e9))


//    println("Start to create model Naive Bayes")
//    interestsStartTime = System.nanoTime()
//    NaiveBayerPredictModel(spark, df3, sc)
//
//    println("Model created, elapsed time: "+ ((System.nanoTime() - interestsStartTime) / 1e9))

    spark.stop()
    //spark.close()

  }
}
