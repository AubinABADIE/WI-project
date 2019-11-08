import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.ml.feature.{VectorAssembler}

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import scala.collection.immutable.HashMap
import org.apache.spark.sql.functions._

import scala.collection.mutable
//import org.apache.spark.ml.feature.VectorAssembler
//import org.apache.spark.ml.linalg.Vectors
//import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
//import org.apache.spark.ml.classification.NaiveBayes
//import org.apache.spark.ml.feature.StringIndexer
//import org.apache.spark.mllib

import NaiveBayesCleaning._
import NaiveBayesPredictModel._

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

    val interests = spark
      .read
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .csv("data/interests.csv")

    val interestsCodes = interests
      .as[(String, String)]
      .collect
      .toMap


    /**
      * Create interests name
      */
    val codeToInterestUDF = udf((interestsArray: mutable.WrappedArray[String]) => {
      if (interestsArray == null) mutable.WrappedArray.empty[String]
      else {
        interestsArray.map((code: String) => {
          if (!code.startsWith("IAB")) code.toLowerCase //Not an interest code
          else interestsCodes(code).toLowerCase
        })
      }
    }).apply(col("interests"))


//    //Converting the interests column to an array of interests instead of interests separated by ","
//    val df3 = df2
//      .withColumn("interests", split($"interests", ",").cast("array<String>"))
//
//    //Converting interest codes to their names and sorting them to avoid redundant data
//    val df4 = df3
//      .withColumn("interestsAsNames", array_distinct(codeToInterestUDF))

    //val mediasMap = df4
    val mediasMap = df
      .select("media")
      .distinct()
      .collect()
      .map(element => element.get(0))
      .zip(Stream from 1)
      .map(mediaTuple => mediaTuple._1 -> mediaTuple._2.toDouble)
      .toMap

    //val df5 = df4
    val df3 = df
      .withColumn("media", udf((elem: String) => mediasMap(elem)).apply(col("media")))

//    val interestIndex = df5
//      .select("interestsAsNames")
//      .distinct()
//      .collect()
//      .flatMap(element => element.get(0).asInstanceOf[mutable.WrappedArray[String]])
//      .distinct
//      .zip(Stream from 1)
//      .map(interests => interests._1 -> interests._2)
//      .toMap
//
//    val df6 = df5
//      .withColumn("interestsIndex", udf((elem: mutable.WrappedArray[String]) =>
//        elem.map(e => interestIndex(e))).apply(col("interestsAsNames")))

    val df7 = clean(spark, df3)

    val df8 = df7.drop(col("size")).drop(col("interests"))

    df8.show()



    // Keep only features and label
    val featuresCols = Array("appOrSite", "city", "media", "network", "os", "publisher", "type", "user") //size, interests
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")


    //omg this is terrible
    val labeled = assembler.transform(df8).rdd.map(row =>{
      val denseVector = row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense
      LabeledPoint(
        row.getDouble(2), //2 is the index of label
        org.apache.spark.mllib.linalg.Vectors.fromML(denseVector)
      )
    }
    )

    val Array(trainingData, test) = labeled.randomSplit(Array(0.8, 0.2))


    //trainingData.collect().take(10).foreach(println(_))


    //NaiveBayes model
    val model = NaiveBayes.train(trainingData)//, lambda = 1.0, modelType = "multinomial")
    val predictedClassification = test.map( x => (model.predict(x.features), x.label))
    val metrics = new MulticlassMetrics(predictedClassification)

    val confusionMatrix = metrics.confusionMatrix
    println("Confusion Matrix",confusionMatrix.asML)

    val myModelStat=Seq(metrics.accuracy,metrics.weightedPrecision,metrics.weightedFMeasure,metrics.weightedRecall)

    myModelStat.foreach(println(_))



    //    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
//    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    //print("Accuracy:" + accuracy)// without size and interests Accuracy:0.615340624059396

    // Save and load model
//    model.save(sc, "data/myNaiveBayesModel")
//    val sameModel = NaiveBayesModel.load(sc, "data/myNaiveBayesModel")


    //NaiveBayerPredictModel(spark, df8, sc)




    spark.stop()
    //spark.close()

  }
}
