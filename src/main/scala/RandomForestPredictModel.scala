import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.RandomForest

object RandomForestPredictModel {
  def RandomForestPredictModel(spark: SparkSession, dataFrame: DataFrame, sc : SparkContext) : Unit = {
    import spark.implicits._

//    val columns: Array[String] = dataFrame
//      .columns
//      .filterNot(_ == "label")
//
//    val finalDFAssembler = new VectorAssembler()
//      .setInputCols(columns)
//      .setOutputCol("features")
//      .setHandleInvalid("skip")
//
//    val finalDF = finalDFAssembler.transform(dataFrame).select( $"features", $"label")
//
//    finalDF.show()
//
//    val datasets = finalDF
//      .randomSplit(Array(0.8,0.2), seed = 42)
//    val training = datasets(0).cache()
//    val testing = datasets(1)


//    //val featuresCols = Array("appOrSite", "city", "media", "network", "os", "publisher", "type", "user")
      val featuresCols = Array("timestamp", "size", "impid", "bidfloor","exchange","appOrSite", "city", "media", "network", "publisher", "type", "i1","i2","i3","i4","i6","i7","i8","i9","i11","i12","i13", "i14","i16","i17","i19","i24","i25","i26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")
      .setHandleInvalid("skip")


    //TestDF I will use later
    val Array(trainDF, testDF) = dataFrame.randomSplit(Array(0.8, 0.2))

    //Stratified sampling
    val trueValue = trainDF.filter("label = true")
    val falseValue = trainDF.filter("label = false").sample(0.030)

    //Equal proportion
    val stratifiedSampling = trueValue.union(falseValue)

    //omg this is terrible
    val labeled = assembler.transform(trainDF).rdd.map(row =>{
      val denseVector = row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense
      LabeledPoint(
        row.getAs[Double]("label"),
        org.apache.spark.mllib.linalg.Vectors.fromML(denseVector)
      )
    }
    )


    val labeledTest = assembler.transform(testDF).rdd.map(row =>{
      val denseVector = row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense
      LabeledPoint(
        row.getAs[Double]("label"),
        org.apache.spark.mllib.linalg.Vectors.fromML(denseVector)
      )
    }
    )

    //val Array(trainingData, test) = labeled.randomSplit(Array(0.7, 0.3))

    //Train forest model
    val treeStrategy = Strategy.defaultStrategy("Classification")

    val numTrees = 10 // Use more in practice.

    val featureSubsetStrategy = "auto" // Let the algorithm choose.

    val model = RandomForest.trainClassifier(labeled, treeStrategy, numTrees, featureSubsetStrategy, seed = 12345)


    // Evaluate model on test instances and compute test error
    val predictedClassification = labeledTest.map( x => (model.predict(x.features), x.label))


    val testErr = labeledTest.map { point =>
      val prediction = model.predict(point.features)


      if (point.label == prediction) 1.0 else 0.0

    }.mean()

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







    //println("Learned Random Forest:n" + model.toDebugString)

    println("Test Error = " + testErr)

  }
}
