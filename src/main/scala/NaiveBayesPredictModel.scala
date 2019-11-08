import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}

object NaiveBayesPredictModel {
  def NaiveBayerPredictModel(spark: SparkSession, dataFrame: DataFrame, sc : SparkContext) : Unit = {
    import spark.implicits._

    //All features must be in double type
    val featuresCols = Array("appOrSite", "city", "media", "network", "os", "publisher", "type", "user") //size, interests
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    // Keep only features and label
    val labeled = assembler.transform(dataFrame).rdd.map(row => LabeledPoint(
      row.getAs[Double]("label"),
      row.getAs[org.apache.spark.mllib.linalg.Vector]("features")
    ))

    print(labeled)

    val Array(trainingData, test) = labeled.randomSplit(Array(0.8, 0.2))


    val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    // Save and load model
    model.save(sc, "data/myNaiveBayesModel")
    val sameModel = NaiveBayesModel.load(sc, "data/myNaiveBayesModel")
  }
}
