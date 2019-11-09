import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

object NaiveBayesCleaning {
  /**
    * For Naive Bayes all values must be double
    */

  def clean(spark: SparkSession, dataFrame: DataFrame) : DataFrame = {
    import spark.implicits._

    /**
      * City not located -> 1.0
      * City located -> 2.0
      */
    val city = dataFrame.
      withColumn("city", udf((elem: String) =>
        if(elem == null)
          1.toDouble
        else
          2.toDouble
      )
        .apply(col("city")))

    /**
      *
      * site -> 1.0
      * app -> 2.0
      *
      */
    val appOrSite = city.
      withColumn("appOrSite", udf((elem: String) =>
        if (elem == "site")
          1.toDouble
        else
          2.toDouble
      ).apply(col("appOrSite")))

    /**
      * Label to double
      *
      * 0.0 -> false
      * 1.0 -> true
      */
    val label = appOrSite.
      withColumn("label", udf((elem: Boolean) =>
        if(elem)
          1.toDouble
        else
          0.toDouble
      ).apply(col("label")))


    /**
      * Publisher to double
      */
    val publisherMap = label
      .select("publisher")
      .distinct()
      .collect()
      .map(element => element.get(0))
      .zip(Stream from 1)
      .map(mediaTuple => mediaTuple._1 -> mediaTuple._2.toDouble)
      .toMap

    val publisher = label
      .withColumn("publisher", udf((elem: String) => publisherMap(elem)).apply(col("publisher")))


    /**
      * User to double
      */
    val userMap = publisher
      .select("user")
      .distinct()
      .collect()
      .map(element => element.get(0))
      .zip(Stream from 1)
      .map(mediaTuple => mediaTuple._1 -> mediaTuple._2.toDouble)
      .toMap

    val user = publisher
      .withColumn("user", udf((elem: String) => userMap(elem)).apply(col("user")))

    /**
      * Network to double
      */
    val networkMap = user
      .select("network")
      .distinct()
      .collect()
      .map(element => element.get(0))
      .zip(Stream from 1)
      .map(mediaTuple => mediaTuple._1 -> mediaTuple._2.toDouble)
      .toMap

    val network = user
      .withColumn("network", udf((elem: String) => networkMap(elem)).apply(col("network")))

    /**
      * Type to double
      */
    val typeMap = network
      .select("type")
      .distinct()
      .collect()
      .map(element => element.get(0))
      .zip(Stream from 1)
      .map(mediaTuple => mediaTuple._1 -> mediaTuple._2.toDouble)
      .toMap

    val typeCol = network
      .withColumn("type", udf((elem: String) => typeMap(elem)).apply(col("type")))

    //OS names cleaning
    /**
      * Android -> 1.0
      * iOS -> 2.0
      * windows -> 3.0
      * Other -> 4.0
      */
    val osRename = typeCol.withColumn("os", udf((elem: String) => {
      elem match {
        case null => "other"
        case x if x.toLowerCase().contains("android") => "android"
        case x if x.toLowerCase().contains("ios") => "ios"
        case x if x.toLowerCase().contains("windows") => "windows"
        case _ => "other"
      }
    }).apply(col("os")))

    val osMap = osRename
      .select("os")
      .distinct()
      .collect()
      .map(element => element.get(0))
      .zip(Stream from 1)
      .map(mediaTuple => mediaTuple._1 -> mediaTuple._2.toDouble)
      .toMap

    val os = osRename
      .withColumn("os", udf((elem: String) => osMap(elem)).apply(col("os")))


    /**
      * Media cleaning to Naive Bayes
      */
    val mediasMap = os
      .select("media")
      .distinct()
      .collect()
      .map(element => element.get(0))
      .zip(Stream from 1)
      .map(mediaTuple => mediaTuple._1 -> mediaTuple._2.toDouble)
      .toMap

    val media = os
      .withColumn("media", udf((elem: String) => mediasMap(elem)).apply(col("media")))

    /**
      * Exchange cleaning to Naive Bayes
      */
    val exchangeMap = media
      .select("exchange")
      .distinct()
      .collect()
      .map(element => element.get(0))
      .zip(Stream from 1)
      .map(mediaTuple => mediaTuple._1 -> mediaTuple._2.toDouble)
      .toMap

    val exchange = media
      .withColumn("exchange", udf((elem: String) => exchangeMap(elem)).apply(col("exchange")))

    exchange


  }
}
