package models

import org.apache.spark.sql.{DataFrame, SparkSession}
import etl.Etl

object TrainingModels {

    def trainModels(spark: SparkSession): Unit = {

        import spark.implicits._ // << add this after defining spark

        // Read the datas from csv file
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

        // Read the interests ID from csv file
        val interests = spark
            .read
            .option("header", "true")
            .option("delimiter", ",")
            .option("inferSchema", "true")
            .csv("data/interests.csv")

        // Clean datas
        val datas = Etl.apply(df, interests)

        // Apply the decision tree classifier
        
        /// Try to save it in a csv file ///
    }
}