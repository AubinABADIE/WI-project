import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.sql.functions.{col, udf}

object RandomForestPredictModel {
  def RandomForestPredictModel(spark: SparkSession, dataFrame: DataFrame, sc : SparkContext) : Unit = {
    import spark.implicits._

    //val featuresCols = Array("timestamp", "size", "impid", "bidfloor","exchange","appOrSite", "city", "media", "network", "publisher", "type", "i1","i2","i3","i4","i6","i7","i8","i9","i11","i12","i13", "i14","i16","i17","i19","i24","i25","i26")
    val featuresCols = Array("timestamp", "size", "appOrSite", "city", "media", "network", "type", "os", "i1","i2","i3","i4","i5", "i6","i7","i8","i9","i11","i12","i13", "i14","i16","i17","i19","i24","i25","i26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")
      //.setHandleInvalid("skip")

    //TestDF I will use later
    val fractions = Map(1.0 -> 0.5, 0.0 -> 0.5)
    val Array(trainDF, testDF) = dataFrame.stat.sampleBy("label", fractions, 36L).randomSplit(Array(0.8, 0.2))

    //Stratified sampling
    val trueValue = trainDF.filter("label = true")
    val falseValue = trainDF.filter("label = false").sample(0.030)

    //Equal proportion
    val stratifiedSampling = trueValue.union(falseValue)

    val labeled = assembler.transform(stratifiedSampling).rdd.map(row =>{
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

    //Train forest model
    val treeStrategy = Strategy.defaultStrategy("Classification")

    val numTrees = 64 // Use more in practice.

    val featureSubsetStrategy = "auto" // Let the algorithm choose.

    val model = RandomForest.trainClassifier(labeled, treeStrategy, numTrees, featureSubsetStrategy, seed = 12345)

    model.save(sc, "data/randomForestModel")

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

//    predictedClassification.toDF("predict", "label")
//      .withColumn("label", udf((elem: Double) => {
//        if(elem == 1.0) true
//        else false
//      }).apply(col("label")))
//      .withColumn("predict", udf((elem: Double) => {
//        if(elem == 1.0) true
//        else false
//      }).apply(col("predict")))
//      .coalesce(1) //So just a single part- file will be created
//      .write.mode(SaveMode.Overwrite)
//      .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
//      .option("header","true") //Write the header
//      .csv("data/prediction")

  }
}
