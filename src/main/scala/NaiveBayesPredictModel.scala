import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}

object NaiveBayesPredictModel {
  def NaiveBayerPredictModel(spark: SparkSession, dataFrame: DataFrame, sc : SparkContext) : Unit = {
    import spark.implicits._

    //val featuresCols = Array("appOrSite", "city", "media", "network", "os", "publisher", "type", "user")
    val featuresCols = Array("exchange","appOrSite", "city", "media", "network", "os", "publisher", "type", "user", "i1","i2","i3","i4","i5","i6", "i7","i8","i9","i10","i11","i12","i13", "i14","i15","i16","i17","i18","i19", "i20","i21","i22","i23","i24","i25","i26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    /**
      * Get original dataset and then split it 80% train 20% test
      *
      * Use 80% but with equal quantity of true/false proportion
      * Train the model
      *
      * Test the model with the 20% from original dataset
      */

    //TestDF I will use later
    val Array(trainDF, testDF) = dataFrame.randomSplit(Array(0.8, 0.2))

    //Stratified sampling
    val trueValue = trainDF.filter("label = true")
    val falseValue = trainDF.filter("label = false").sample(0.030)

    //Equal proportion
    val stratifiedSampling = trueValue.union(falseValue)

    //omg this is terrible
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

    //val Array(trainingData, test) = labeled.randomSplit(Array(0.8, 0.2))
    //trainingData.collect().take(10).foreach(println(_))

    //NaiveBayes model
//    val model = NaiveBayes.train(trainingData)//, lambda = 1.0, modelType = "multinomial")
//    val predictedClassification = test.map( x => (model.predict(x.features), x.label))

    val model = NaiveBayes.train(labeled)//, lambda = 1.0, modelType = "multinomial")
    val predictedClassification = labeledTest.map( x => (model.predict(x.features), x.label))
    val metrics = new MulticlassMetrics(predictedClassification)
//    val metrics = new BinaryClassificationMetrics(predictedClassification)


//    // Precision by threshold
//    val precision = metrics.precisionByThreshold
//    precision.foreach { case (t, p) =>
//      println(s"Threshold: $t, Precision: $p")
//    }
//
//    // Recall by threshold
//    val recall = metrics.recallByThreshold
//    recall.foreach { case (t, r) =>
//      println(s"Threshold: $t, Recall: $r")
//    }
//
//    // Precision-Recall Curve
//    val PRC = metrics.pr
//
//    // F-measure
//    val f1Score = metrics.fMeasureByThreshold
//    f1Score.foreach { case (t, f) =>
//      println(s"Threshold: $t, F-score: $f, Beta = 1")
//    }
//
//    val beta = 0.5
//    val fScore = metrics.fMeasureByThreshold(beta)
//    f1Score.foreach { case (t, f) =>
//      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
//    }
//
//    // AUPRC
//    val auPRC = metrics.areaUnderPR
//    println("Area under precision-recall curve = " + auPRC)
//
//    // Compute thresholds used in ROC and PR curves
//    val thresholds = precision.map(_._1)
//
//    // ROC Curve
//    val roc = metrics.roc
//
//    // AUROC
//    val auROC = metrics.areaUnderROC
//    println("Area under ROC = " + auROC)
//    val confusionMatrix = metrics.confusionMatrix
//    println("Confusion Matrix",confusionMatrix.asML)
//
//    val myModelStat=Seq(metrics.accuracy,metrics.weightedPrecision,metrics.weightedFMeasure,metrics.weightedRecall)
//
//    myModelStat.foreach(println(_))
//
//    metrics.labels.foreach(println(_))
    //println("Labels" + metrics.labels.)


    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")


  }
}
