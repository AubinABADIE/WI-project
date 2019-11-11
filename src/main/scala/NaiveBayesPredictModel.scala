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

    val featuresCols = Array("timestamp", "size", "impid", "bidfloor","exchange","appOrSite", "city", "media", "network", "publisher", "type", "i1","i2","i3","i4","i6", "i7","i8","i9","i11","i12","i13", "i14","i16","i17","i19","i24","i25","i26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")
      .setHandleInvalid("skip")

    /**
      * Get original dataset and then split it 80% train and 20% test
      *
      * Use 80% but with equal quantity of true/false proportion
      * Train the model
      *
      * Test the model with the 20% from original dataset
      */

    //TestDF will be used to test
    val Array(trainWithoutProportion, testWithoutProportion) = dataFrame.randomSplit(Array(0.8, 0.2))

    //Stratified sampling
    val trueValue = trainWithoutProportion.filter("label = true")
    val falseValue = trainWithoutProportion.filter("label = false").sample(0.030)

    //Equal proportion
    val sameProportionSampleTrain = trueValue.union(falseValue)

    //omg this is terrible
    val featuresLabelProportionTrain = assembler.transform(trainWithoutProportion).rdd.map(row =>{
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

//    println("Starting to save Naive Bayes model")
//    val modelStartTime = System.nanoTime()
//    model.save(sc, "data/naiveModel")
//    val elapsedTimeModel = (System.nanoTime() - modelStartTime) / 1e9
//    println("[DONE] Naive Bayes model, elapsed time: "+(elapsedTimeModel - (elapsedTimeModel % 0.01))+"min")



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

    predictedClassification.toDF("predict", "label")
      .withColumn("label", udf((elem: Double) => {
        if(elem == 1.0) true
        else false
    }).apply(col("label")))
      .withColumn("predict", udf((elem: Double) => {
        if(elem == 1.0) true
        else false
      }).apply(col("predict")))
      .coalesce(1) //So just a single part- file will be created
          .write.mode(SaveMode.Overwrite)
          .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
          .option("header","true") //Write the header
          .csv("data/prediction")

    //predictedClassification.saveAsTextFile("data/filename.csv")

//    // write to file:
//    new PrintWriter("filename.csv") { write(csv); close() }

//    predictedClassification
//      .coalesce(1)
//      .write
//      .option("header","true")
//      .mode("overwrite")
//      .csv("prediction")

  }
}
