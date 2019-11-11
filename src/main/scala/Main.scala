<<<<<<< HEAD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.log4j.{Level, Logger}
import models._

=======
import java.nio.file.{Files, Paths}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import NaiveBayesCleaning._
import NaiveBayesPredictModel._
import IntererestsCleaning._
import LogisticRegressionModel._
>>>>>>> naiveBayes
object Main extends App {

  override def main(args: Array[String]) {

    val start = System.nanoTime()
    var jsonFile = "data-students.json"
    var model = "lr" //rf

    val message = "Usage: ./clickPredict jsonFile [option]\nOptions: \n\t* Naive Bayes --naive\n\t* Logistic Regression --lr \n\t* Random Forest --rf\n\nExample ./clickPredict data-students.json naive"

    if(args.isEmpty) println(message)
    if(args.length == 2){
      jsonFile = args(0)
      model = args(1)
    }else if (args.length == 1){
      jsonFile = args(0)
    }else{
      println(message)
    }

    if(!Files.exists(Paths.get("data/"+jsonFile))){
      println("File "+ jsonFile + "does not exists")
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

<<<<<<< HEAD
    TrainingModels.trainModels(spark)

    spark.close()
=======
    import spark.implicits._ // << add this after defining spark

    val df = spark
      .read
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .json("data/"+jsonFile)
      //.csv("data/exportCSVANOVA")
      //.drop("interests")
      //.limit(1000)

    println("Starting to generate interests")
    var interestsStartTime = System.nanoTime()

    val dfInterests = cleanInterests(spark, df)

    val elapsedTimeInterests = (System.nanoTime() - interestsStartTime) / 1e9

    println("Interests DONE, elapsed time: "+(elapsedTimeInterests - (elapsedTimeInterests % 0.01))+"s")

    //Drop generated ID column
    val df2 = dfInterests.drop(col("id"))

    println("Starting to clean dataset")
    val cleaningStartTime = System.nanoTime()

    val df3 = clean(spark, df2)

    val elapsedTimeCleaning = (System.nanoTime() - cleaningStartTime) / 1e9
    println("Cleaning DONE, elapsed time: "+ (elapsedTimeCleaning - (elapsedTimeCleaning % 0.01))+"s")

    /**
      * After ANOVA drop the following columns
      * os, user, i5, i10, i15, i18, i20, i21, i22, i23
      */
    val df4 = df3.drop("os", "user", "i5", "i10", "i15", "i18", "i20", "i21", "i22", "i23", "interests")

//    println("Start to export CSV")
//    interestsStartTime = System.nanoTime()
//
//    df4.coalesce(1) //So just a single part- file will be created
//      .write.mode(SaveMode.Overwrite)
//      .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
//      .option("header","true") //Write the header
//      .csv("data/exportCSVANOVA")
//
//    println("Export DONE, elapsed time: "+ (elapsedTimeInterests - (elapsedTimeInterests % 0.01))+"ms")

    if(model == "naive"){
      println("Start to create model Naive Bayes")
      val modelStartTime = System.nanoTime()
      NaiveBayesPredictModel.NaiveBayerPredictModel(spark, df4, sc)
      val elapsedTimeModel = (System.nanoTime() - modelStartTime) / 1e9
      println("Naive Bayes model created, elapsed time: "+ (elapsedTimeModel - (elapsedTimeModel % 0.01))+"ms")
    }else if(model == "lr"){


      val modelStartTime = System.nanoTime()
      LogisticRegressionModel.LogisticRegressionModel(spark, df4, sc)
      val elapsedTimeModel = (System.nanoTime() - modelStartTime) / 1e9
      println("Logistic regression model created, elapsed time: "+ (elapsedTimeModel - (elapsedTimeModel % 0.01))+"ms")
      //RandomForestPredictModel.RandomForestPredictModel(spark, df, sc)
    }else if(model == "rf"){
      val modelStartTime = System.nanoTime()
      RandomForestPredictModel.RandomForestPredictModel(spark, df4, sc)
      val elapsedTimeModel = (System.nanoTime() - modelStartTime) / 1e9
      println("Random forest model created, elapsed time: "+ (elapsedTimeModel - (elapsedTimeModel % 0.01))+"ms")
    }

    val elapedStime = (System.nanoTime() - start) / 1e9
    println("End: "+ elapedStime)



    spark.stop()
>>>>>>> naiveBayes
  }
}
