import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.log4j.{Level, Logger}
import scala.collection.immutable.HashMap
import org.apache.spark.sql.functions._

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

    val df = spark
      .read
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .json("data/data-students.json")
    import spark.implicits._ // << add this


    //df.show(20)

    /**
     * A blessed alternative => using a hash map and literally map the values on the
     *  dataframe to a new value. I couldnt gat it to work (missing encoders?) but give it
     *  a shot if you think you can!
     *
     val osNames = HashMap(
                     "Android" -> "android",
                     "iOS" -> "ios",
                     "WindowsMobile" -> "windows",
                     "WindowsPhone" -> "windows",
                     "Windows Phone OS" -> "windows",
                     "Windows Mobile OS" -> "windows")
     */


    val df2 = df.filter(row => row.getAs("os") != null
                                          && row.getAs("os") != "other"
                                          && row.getAs("os") != "Unknown")
                .withColumn("os", when($"os" === "Android", "android")
                                            .otherwise(when($"os" === "iOS", "ios")
                                                .otherwise(when($"os" === "WindowsMobile", "windows")
                                                  .otherwise(when($"os" === "WindowsPhone", "windows")
                                                    .otherwise(when($"os" === "Windows Phone OS", "windows")
                                                      .otherwise(when($"os" === "Windows Mobile OS", "windows")
                                                        .otherwise($"os")))))))

    df2.select("os").distinct().show()
    df2.show(10)
    spark.close()
  }
}