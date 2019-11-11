import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.functions._

object ETL {
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


    /**
      * timestamp cleaning to Naive Bayes
      *
      * 0.0 => night 0-6
      * 1.0 => morning 6-12
      * 2.0 => afternoon 13-17
      * 3.0 => evening 18-0
      *
      */
    val timestamp = exchange
      .withColumn("Time", to_timestamp(from_unixtime(col("Timestamp"))))
      .withColumn("Date", date_format(col("Time"), "yyyy-MM-dd"))
      .withColumn("EventTime", date_format(col("Time"), "HH:mm:ss"))
      .withColumn("Hour", date_format(col("EventTime"), "HH"))
      .withColumn("timestamp", udf((elem: String) => {
        elem.toInt match {
          case x if 0 until 6 contains x => 0.0
          case x if 6 until 12 contains x => 1.0
          case x if 12 until 17 contains x => 3.0
          case x if 17 until 24 contains x => 4.0
          case _ => 5.0
        }
      }).apply(col("Hour"))).drop("Time").drop("EventTime").drop("Hour").drop("Date")

    /**
      * impid cleaning to Naive Bayes
      */
    val impidMap = timestamp
      .select("impid")
      .distinct()
      .collect()
      .map(element => element.get(0))
      .zip(Stream from 1)
      .map(mediaTuple => mediaTuple._1 -> mediaTuple._2.toDouble)
      .toMap

    val impid = timestamp
      .withColumn("impid", udf((elem: String) => impidMap(elem)).apply(col("impid")))


    val size1 = impid.withColumn("size", col("size").cast("String"))
    val size = size1.withColumn("size", udf((elem: String) =>

      elem match {
        case "[380, 214]" => 0.0
        case "[358, 201]" => 0.0
        case "[343, 193]" => 0.0
        case "[288, 162]" => 0.0
        case "[291, 164]" => 0.0
        case "[300, 169]" => 0.0
        case "[569, 320]" => 0.0
        case "[600, 350]" => 0.0
        case "[328, 185]" => 0.0
        case "[640, 360]" => 0.0
        case "[306, 172]" => 0.0
        case "[1280, 720]" => 0.0
        case "[320, 480]" => 1.0
        case "[325, 183]" => 2.0
        case "[320, 50]" => 2.0
        case "[320, 250]" => 2.0
        case "[300, 250]" => 2.0
        case "[480, 320]" => 3.0
        case "[610, 390]" => 3.0
        case "[500, 300]" => 4.0
        case "[768, 1024]" => 5.0
        case _ => 6.0
      }
    ).apply(col("size")))

    size
  }
}
