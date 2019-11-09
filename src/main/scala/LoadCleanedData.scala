import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import NaiveBayesPredictModel._
import RandomForestPredictModel._
import org.apache.spark.sql.types.DoubleType

object LoadCleanedData extends App {


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
    .csv("data/exportComplete.csv")

  val cols = df.columns
  val result = cols.indices.foldLeft(df) {
    (newDF, i) => newDF.withColumn(cols(i), newDF(cols(i)).cast(DoubleType))
  }

  result.show()

  //NaiveBayerPredictModel(spark, result, sc)
  RandomForestPredictModel.RandomForestPredictModel(spark, result, sc)


  spark.close()
}
