package models

import org.apache.spark.sql
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint

object DecisionTreeMod {

    def method(df: DataFrame, spark: SparkSession): DecisionTreeModel = {

        import spark.implicits._

        val d: RDD[Row] = df.select("features", "labelIndex").withColumnRenamed("labelIndex", "label").rdd
        //transform row into labeledPoint
        val data: RDD[LabeledPoint] = d.map((l: Row) => LabeledPoint(l.getAs(1), Vectors.fromML(l.getAs(0)).toDense))

        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 42)

        // Train a DecisionTree model.
        //  Empty categoricalFeaturesInfo indicates all features are continuous.
        val numClasses = 2
        val categoricalFeaturesInfo = Map[Int, Int]()
        val impurity = "gini"
        val maxDepth = 2
        val maxBins = 607491

        val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

        // Evaluate model on test instances and compute test error
        val predictWithLabels = testData.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
        }

        metrics(predictWithLabels)

        return model
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