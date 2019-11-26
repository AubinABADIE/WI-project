import java.io.PrintWriter

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.functions.{col, udf}

object NaiveBayesPredictModel {
  def NaiveBayerPredictModel(spark: SparkSession, dataFrame: DataFrame, sc : SparkContext) : Unit = {
    import spark.implicits._

    val columns: Array[String] = dataFrame
      .columns
      .filterNot(_ == "label")
      .filterNot(_ == "impid")
      .filterNot(_ == "bidfloor")
      .filterNot(_ == "exchange")
      .filterNot(_ == "publisher")
    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")
      //.setHandleInvalid("keep")



    /**
      * Get original dataset and then split it 80% train and 20% test
      *
      * Use 80% but with equal quantity of true/false proportion
      * Train the model
      *
      * Test the model with the 20% from original dataset
      */

    //TestDF will be used to test
//    val Array(trainWithoutProportion, testWithoutProportion) = dataFrame.randomSplit(Array(0.8, 0.2))
    val fractions = Map(1.0 -> 1.0, 0.0 -> 1.0)
    val Array(trainWithoutProportion, testWithoutProportion) = dataFrame.stat.sampleBy("label", fractions, 36L).randomSplit(Array(0.8, 0.2))

    System.out.println(trainWithoutProportion.count() + "where " + trainWithoutProportion.filter("label = true").count() + " is true")
    System.out.println(testWithoutProportion.count() + "where " + testWithoutProportion.filter("label = true").count() + " is true")


    //Stratified sampling
    val trueValue = trainWithoutProportion.filter("label = true")
    val falseValue = trainWithoutProportion.filter("label = false").sample(0.030)

    //Equal proportion
    val sameProportionSampleTrain = trueValue.union(falseValue)

    val featuresLabelProportionTrain = assembler.transform(sameProportionSampleTrain).rdd.map(row =>{
      val denseVector = row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense
      LabeledPoint(
        row.getAs[Double]("label"),
        org.apache.spark.mllib.linalg.Vectors.fromML(denseVector)
      )
    }
    )


    val featuresLabelWithoutProportiondTest = assembler.transform(testWithoutProportion).rdd.map(row =>{
      val denseVector = row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense
      LabeledPoint(
        row.getAs[Double]("label"),
        org.apache.spark.mllib.linalg.Vectors.fromML(denseVector)
      )
    }
    )



    val model = NaiveBayes.train(featuresLabelProportionTrain)



    val predictedClassification = featuresLabelWithoutProportiondTest.map( x => (model.predict(x.features), x.label))

    println("Starting to save Naive Bayes model")
    val modelStartTime = System.nanoTime()
    model.save(sc, "data/naiveModel")
    val elapsedTimeModel = (System.nanoTime() - modelStartTime) / 1e9
    println("[DONE] Naive Bayes model, elapsed time: "+(elapsedTimeModel - (elapsedTimeModel % 0.01))+"min")



    val metrics = new MulticlassMetrics(predictedClassification)
    val bmetrics = new BinaryClassificationMetrics(predictedClassification)

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
