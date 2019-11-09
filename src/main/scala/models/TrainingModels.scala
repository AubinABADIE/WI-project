package models

import etl.Etl

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql
import org.apache.spark.sql.{Row, DataFrame, SparkSession}

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
      .drop("size")

    // Read the interests ID from csv file
    val interests = spark
      .read
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .csv("data/interests.csv")

    // Clean datas
    val datas = Etl.apply(df, interests)
    logisticRegression(datas, spark)
  }

  def logisticRegression(df: DataFrame, spark: SparkSession): LogisticRegressionModel = {
    import spark.implicits._
    //Select features and label as resilient distributed dataset
    val data = df
      .select("features", "labelIndex")
      .withColumnRenamed("labelIndex", "label")

    //Splits RDD into two datasets, one containing 80% of data for training and another one containing 20% for testing
    val datasets = data
      .randomSplit(Array(0.8,0.2), seed = 42)
    val training = datasets(0).cache()
    val testing = datasets(1)

    //Model creation
    val model = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(100)
      .fit(training)

    // println(s"Coefficients: ${model.coefficients} \nIntercept: ${model.intercept}")

    val predict = model.transform(testing)
    val predictWithLabels = predict
      .select("prediction", "label")
      .as[(Double, Double)]
      .rdd

    metrics(predictWithLabels)

    model
  }

  def metrics(predictionWithLabels: RDD[(Double, Double)]) = {
    val metrics = new MulticlassMetrics(predictionWithLabels)
    val bmetrics = new BinaryClassificationMetrics(predictionWithLabels)
    //accuracy
    val accuracy = metrics.accuracy
    println(s"Accuracy = $accuracy")

    //precision and recall
    val labels = metrics.labels
    labels.foreach(label => {
      println(s"Precision($label) = " + metrics.precision(label))
      println(s"Recall($label) = " + metrics.recall(label))
    })

    //confusion matrix
    println("Confusion matrix:\n" + metrics.confusionMatrix)

    //Area under ROC
    val auROC = bmetrics.areaUnderROC
    println(s"Area under ROC: $auROC")

    metrics
  }
}