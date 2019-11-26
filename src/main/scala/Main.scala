import java.nio.file.{Files, Paths}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import IntererestsCleaning._
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
object Main extends App {

  override def main(args: Array[String]) {

    val start = System.nanoTime()
    var jsonFile = "data-students.json"
    var model = "rf" //rf

    val message = "Usage: run jsonFile [option]\nOptions: \n\t* Naive Bayes --naive\n\t* Logistic Regression --lr \n\t* Random Forest --rf\n\nExample run data-students.json naive"

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

    import spark.implicits._ // << add this after defining spark

    val df = spark
      .read
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .json("data/"+jsonFile)
      //.csv("data/exportCSV")
      //.limit(10)

    println("Starting to generate interests")
    val interestsStartTime = System.nanoTime()

    val dfInterests = cleanInterests(spark, df)

    val elapsedTimeInterests = (System.nanoTime() - interestsStartTime) / 1e9

    println("Interests DONE, elapsed time: "+(elapsedTimeInterests - (elapsedTimeInterests % 0.01))+"s")

    //Drop generated ID column
    val df2 = dfInterests.drop(col("id"))

    println("Starting to clean dataset")
    val cleaningStartTime = System.nanoTime()

    val df3 = ETL.clean(spark, df2)

    val elapsedTimeCleaning = (System.nanoTime() - cleaningStartTime) / 1e9
    println("Cleaning DONE, elapsed time: "+ (elapsedTimeCleaning - (elapsedTimeCleaning % 0.01))+"s")

    /**
      * After ANOVA drop the following columns
      * user
      */
    val df4 = df3.drop("user", "interests")

    /**
      * Generate CSV to faster training
      */
//    println("Start to export CSV")
//    interestsStartTime = System.nanoTime()
//    df4.coalesce(1) //So just a single part- file will be created
//      .write.mode(SaveMode.Overwrite)
//      .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
//      .option("header","true") //Write the header
//      .csv("data/exportCSV")
//    println("Export DONE, elapsed time: "+ (elapsedTimeInterests - (elapsedTimeInterests % 0.01))+"ms")

    /**
      * Generate models
      */

//    val df4 = df
//    RandomForestPredictModel.RandomForestPredictModel(spark, df4, sc)
//    LogisticRegressionPredictModel.LogisticRegressionPredictModel(spark, df4, sc)
//    NaiveBayesPredictModel.NaiveBayerPredictModel(spark, df4, sc)
//    sys.exit()

    if(model == "naive"){
      println("Start to create predictions with Naive Bayes model")
      val modelStartTime = System.nanoTime()
      val predictModel = NaiveBayesModel.load(sc, "data/naiveModel")
      val columns: Array[String] = df4
        .columns
        .filterNot(_ == "label")
        .filterNot(_ == "impid")
        .filterNot(_ == "bidfloor")
        .filterNot(_ == "exchange")
        .filterNot(_ == "publisher")

      val finalDFAssembler = new VectorAssembler()
        .setInputCols(columns)
        .setOutputCol("features")
        .setHandleInvalid("keep")

      val labeled = finalDFAssembler.transform(df4).rdd.map(row =>{
        val denseVector = row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense
        LabeledPoint(
          row.getAs[Double]("label"),
          org.apache.spark.mllib.linalg.Vectors.fromML(denseVector)
        )
      }
      )

      val prediction = labeled.map( x => (predictModel.predict(x.features), x.label))

      val withoutLabel = df.drop("label").drop("size")

      prediction.toDF("predict", "label").withColumn("id", monotonically_increasing_id())
        .join(withoutLabel.withColumn("id", monotonically_increasing_id()), Seq("id"))
        .drop("id")
        .withColumn("label", udf((elem: Double) => {
          if(elem == 1.0) true
          else false
        }).apply(col("predict")))
        .drop("predict")
        .coalesce(1) //So just a single part- file will be created
        .write.mode(SaveMode.Overwrite)
        .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
        .option("header","true") //Write the header
        .csv("data/prediction")

      val elapsedTimeModel = (System.nanoTime() - modelStartTime) / 1e9
      println("Prediction done, CSV exported, elapsed time: "+ (elapsedTimeModel - (elapsedTimeModel % 0.01))+"s")
    }else if(model == "lr"){
      println("Start to create predictions with Logistic Regression model")
      val modelStartTime = System.nanoTime()

      val predictModel = LogisticRegressionModel.load("data/LRModel")

      val columns: Array[String] = df4
        .columns
        .filterNot(_ == "label")
        .filterNot(_ == "impid")
        .filterNot(_ == "bidfloor")
        .filterNot(_ == "exchange")
        .filterNot(_ == "publisher")

      val finalDFAssembler = new VectorAssembler()
        .setInputCols(columns)
        .setOutputCol("features")
      //.setHandleInvalid("keep")

      val finalDF = finalDFAssembler.transform(df4).select( $"features", $"label")

      val predict = predictModel.transform(finalDF)
      val predictWithLabels = predict
        .select("prediction", "label")
        .as[(Double, Double)]
        .rdd

      val withoutLabel = df.drop("label").drop("size")

      predictWithLabels.toDF("predict", "label").withColumn("id", monotonically_increasing_id())
        .join(withoutLabel.withColumn("id", monotonically_increasing_id()), Seq("id"))
        .drop("id")
        .withColumn("label", udf((elem: Double) => {
          if(elem == 1.0) true
          else false
        }).apply(col("predict")))
        .drop("predict")
        .coalesce(1) //So just a single part- file will be created
        .write.mode(SaveMode.Overwrite)
        .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
        .option("header","true") //Write the header
        .csv("data/prediction")

      //LogisticRegressionPredictModel.LogisticRegressionPredictModel(spark, df4, sc)
      val elapsedTimeModel = (System.nanoTime() - modelStartTime) / 1e9
      println("Logistic regression predictions created, elapsed time: "+ (elapsedTimeModel - (elapsedTimeModel % 0.01))+"s")
    }else if(model == "rf"){
      println("Start to create predictions with Random Forest model")
      val modelStartTime = System.nanoTime()

      val predictModel = RandomForestModel.load(sc, "data/randomForestModel")

      val columns: Array[String] = df4
        .columns
        .filterNot(_ == "label")

      val finalDFAssembler = new VectorAssembler()
        .setInputCols(columns)
        .setOutputCol("features")
        .setHandleInvalid("keep")

      val labeled = finalDFAssembler.transform(df4).rdd.map(row =>{
        val denseVector = row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense
        LabeledPoint(
          row.getAs[Double]("label"),
          org.apache.spark.mllib.linalg.Vectors.fromML(denseVector)
        )
      }
      )

      val prediction = labeled.map( x => (predictModel.predict(x.features), x.label))

      val withoutLabel = df.drop("label").drop("size")

      prediction.toDF("predict", "label").withColumn("id", monotonically_increasing_id())
        .join(withoutLabel.withColumn("id", monotonically_increasing_id()), Seq("id"))
        .drop("id")
        .withColumn("label", udf((elem: Double) => {
          if(elem == 1.0) true
          else false
        }).apply(col("predict")))
        .drop("predict")
        .coalesce(1) //So just a single part- file will be created
        .write.mode(SaveMode.Overwrite)
        .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
        .option("header","true") //Write the header
        .csv("data/prediction")

      //RandomForestPredictModel.RandomForestPredictModel(spark, df4, sc)
      val elapsedTimeModel = (System.nanoTime() - modelStartTime) / 1e9
      println("Random forest prediction DONE, CSV created, elapsed time: "+ (elapsedTimeModel - (elapsedTimeModel % 0.01))+"s")
    }

    val elapedStime = (System.nanoTime() - start) / 1e9
    println("End: "+ elapedStime)



    spark.stop()
  }
}
