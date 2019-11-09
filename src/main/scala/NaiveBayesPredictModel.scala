import org.apache.spark.{SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.mllib.classification.{NaiveBayes}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object NaiveBayesPredictModel {
  def NaiveBayerPredictModel(spark: SparkSession, dataFrame: DataFrame, sc : SparkContext) : Unit = {
    import spark.implicits._

    val featuresCols = Array("appOrSite", "city", "media", "network", "os", "publisher", "type", "user")
    //val featuresCols = Array("appOrSite", "city", "media", "network", "os", "publisher", "type", "user", "i1","i2","i3","i4","i5","i6", "i7","i8","i9","i10","i11","i12","i13", "i14","i15","i16","i17","i18","i19", "i20","i21","i22","i23","i24","i25","i26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")


    //omg this is terrible
    val labeled = assembler.transform(dataFrame).rdd.map(row =>{
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
  }
}
