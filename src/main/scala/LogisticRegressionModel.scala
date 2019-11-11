import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SparkSession}

object LogisticRegressionModel {
  def LogisticRegressionModel(spark: SparkSession, dataFrame: DataFrame, sc : SparkContext) : Unit = {
    import spark.implicits._

    val columns: Array[String] = dataFrame
      .columns
      .filterNot(_ == "label")

    val finalDFAssembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val finalDF = finalDFAssembler.transform(dataFrame).select( $"features", $"label")

    finalDF.show()

    val datasets = finalDF
      .randomSplit(Array(0.8,0.2), seed = 42)
    val training = datasets(0).cache()
    val testing = datasets(1)

    //Model creation
    val lr = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(1000)
    //.setRegParam(0.01)
    //.setFamily("binomial")
    //.setThreshold(0.5)

    val lrModel = lr.fit(training)

    val predict = lrModel.transform(testing)
    val predictWithLabels = predict
      .select("prediction", "label")
      .as[(Double, Double)]
      .rdd

    val metrics = new MulticlassMetrics(predictWithLabels)
    val bmetrics = new BinaryClassificationMetrics(predictWithLabels)
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
  }

}
