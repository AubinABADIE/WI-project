import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}

object NaiveBayesPredictModel {
  def NaiveBayerPredictModel(spark: SparkSession, dataFrame: DataFrame, sc : SparkContext) : Unit = {
    import spark.implicits._

    val featuresCols = Array("timestamp", "size", "impid", "bidfloor","exchange","appOrSite", "city", "media", "network", "os", "publisher", "type", "user", "i1","i2","i3","i4","i5","i6", "i7","i8","i9","i10","i11","i12","i13", "i14","i15","i16","i17","i18","i19", "i20","i21","i22","i23","i24","i25","i26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")
      .setHandleInvalid("skip")

    /**
      * Get original dataset and then split it 80% train 20% test
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

    model.save(sc, "data/naiveModel")

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
