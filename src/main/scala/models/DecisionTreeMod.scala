package models

import org.apache.spark.sql
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors

object DecisionTreeMod {

    def method(df: DataFrame, spark: SparkSession): DecisionTreeClassifier = {

        import spark.implicits._

        val data = df
            .select("features", "labelIndex")
            .withColumnRenamed("labelIndex", "label")

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        val labelIndexer = new StringIndexer()
            .setInputCol("label")
            .setOutputCol("indexedLabel")
            .fit(data)
        // Automatically identify categorical features, and index them.
        val featureIndexer = new VectorIndexer()
            .setInputCol("features")
            .setOutputCol("indexedFeatures")
            .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
            .fit(data)

        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 42)

        // Train a DecisionTree model.
        val dt = new DecisionTreeClassifier()
            .setLabelCol("indexedLabel")
            .setFeaturesCol("indexedFeatures")

        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
            .setInputCol("prediction")
            .setOutputCol("predictedLabel")
            .setLabels(labelIndexer.labels)

        // Chain indexers and tree in a Pipeline.
        val pipeline = new Pipeline()
            .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

        // Train model. This also runs the indexers.
        val model = pipeline.fit(trainingData)

        // Make predictions.
        val predictions = model.transform(testData)
        val predictWithLabels = predictions
            .select("prediction", "label")
            .as[(Double, Double)]
            .rdd

        metrics(predictWithLabels)

        return dt
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