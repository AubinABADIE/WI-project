import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.log4j.{Level, Logger}
import models._

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

    TrainingModels.trainModels(spark)

    spark.close()
  }
}
