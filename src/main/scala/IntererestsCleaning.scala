import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object IntererestsCleaning {

  def cleanInterests(spark: SparkSession, dataFrame: DataFrame): DataFrame = {
    import spark.implicits._

    /**
      * If 1 contains the interest
      * if 0 doesnt
      * @param iab
      * @param interests
      * @return
      */
    def exists(iab: List[String], interests: String): Double = {
      val interestsComa = interests + ","
      val interestsDash = interests + "-"
      if (iab.exists(interestsComa.contains) || iab.exists(interestsDash.contains)) {
        1.toDouble
      }
      else 0.toDouble
    }

    val interestsDataFrame = dataFrame.select($"interests")
    val interestsAsStr = interestsDataFrame.map(_.toString)

    val iab1 = List("iab1", "arts", "entertainment", "movies","television","books", "literature", "music","celebrity fan/gossip","fine art","humor")
    val iab2 = List("iab2", "automotive", "luxury","motorcycles","crossover","car culture","certified pre-owned","sedan","road-side assistance","off-road vehicles","coupe","wagon","hybrid","vintage cars","auto parts","auto repair","performance vehicles","convertible","trucks & accessories","hatchback","minivan","buying/selling cars","pickup","diesel","electric vehicle")
    val iab3 = List("iab3", "business", "forestry","logistics","agriculture","human resources","biotech/biomedical","metals","green solutions","business software","construction","marketing","advertising","government")
    val iab4 = List("iab4", "careers", "u.s. military","job search","nursing","financial aid","job fairs","telecommuting","scholarships","career advice","career planning","resume writing/advice","college")
    val iab5 = List("iab5", "education", "art history","graduate school","private school","adult education","english as a 2nd language","college administration","educators","college life","education","homework/study tips","distance learning","homeschooling","language learning","special education","studying business")
    val iab6 = List("iab6", "family", "parenting", "parenting kids","babies & toddlers","special needs kids","parenting teens","eldercare","daycare/pre school","pregnancy","family internet","adoption")
    val iab7 = List("iab7", "health", "fitness", "thyroid disease","bipolar disorder","epilepsy","brain tumor","incontinence","arthritis","add","sexuality","diabetes","depression","chronic pain","pediatrics","orthopedics","herbs for health","weight loss","senior health","aids/hiv","dermatology","panic/anxiety disorders","cold & flu","holistic healing","asthma","sleep disorders","women's health","infertility","headaches/migraines","cholesterol","psychology/psychiatry","smoking cessation","ibs/crohn’s disease","deafness","allergies","autism/pdd","chronic fatigue syndrome","substance abuse","cancer","men’s health","heart disease","dental care","gerd/acid reflux","physical therapy","alternative medicine","incest/abuse support","exercise","nutrition")
    val iab8 = List("iab8", "food", "drink", "wine","food allergies","desserts & baking","cuisine-specific","mexican cuisine","cajun/creole","chinese cuisine","american cuisine","vegan","cocktails/beer","health/low-fat cooking","italian cuisine","vegetarian","french cuisine","coffee/tea","dining out","barbecues & grilling","japanese cuisine")
    val iab9 = List("iab9", "hobbies", "interest", "sci-fi & fantasy","investors & patents","stamps & coins","art/technology","chess","painting","board games/puzzles","drawing/sketching","comic books","photography","genealogy","roleplaying games","video & computer games","home recording","radio","freelance writing","bird-watching","cigars","magic & illusion","screenwriting","scrapbooking","guitar","woodworking","beadwork","jewelry making","arts & crafts","candle & soap making","card games","needlework","collecting","getting published")
    val iab10 = List("iab10", "home", "garden", "landscaping","entertaining","remodeling & construction","appliances","home repair","gardening","home theater","interior decorating","environmental safety")
    val iab11 = List("iab11", "law", "politics", "u.s. government resources","politics","commentary","legal issues","immigration")
    val iab12 = List("iab12", "news", "local news","national news","international news")
    val iab13 = List("iab13", "personal finance", "investing","financial news","retirement planning","beginning investing","insurance","tax planning","credit/debt & loans","hedge fund","stocks","options","financial planning","mutual funds")
    val iab14 = List("iab14", "society", "ethnic specific","marriage","weddings","gay life","divorce support","teens","dating","senior living")
    val iab15 = List("iab15", "science", "biology","physics","botany","paranormal phenomena","astrology","geology","weather","chemistry","space/astronomy","geography")
    val iab16 = List("iab16", "pet", "birds","reptiles","veterinary medicine","aquariums","large animals","dogs","cats")
    val iab17 = List("iab17", "sport", "boxing","volleyball","running/jogging","canoeing/kayaking","nascar racing","skateboarding","fly fishing","horses","bodybuilding","rodeo","walking","surfing/body-boarding","mountain biking","auto racing","figure skating","skiing","sailing","cricket","rugby","hunting/shooting","snowboarding","waterski/wakeboard","game & fish","climbing","paintball","power & motorcycles","martial arts","freshwater fishing","inline skating","pro basketball","golf","table tennis/ping-pong","baseball","saltwater fishing","scuba diving","olympics","football","pro ice hockey","tennis","cheerleading","horse racing","swimming","world soccer","bicycling")
    val iab18 = List("iab18", "style", "fashion", "clothing","body art","accessories","fashion","beauty","jewelry")
    val iab19 = List("iab19", "technology", "computing", "mp3/midi","desktop publishing","c/c++","web clip art","palmtops/pdas","antivirus software","graphics software","computer peripherals","visual basic","network security","shareware/freeware","windows","mac support","databases","portable","home video/dvd","computer networking","computer certification","net for beginners","data centers","unix","javascript","email","cell phones","3d graphics","web search","internet technology","entertainment","web design/html","net conferencing","animation","cameras & camcorders","desktop video","pc support","computer reviews","java")
    val iab20 = List("iab20", "travel", "camping","spas","australia & new zealand","canada","mexico & central america","united kingdom","greece","eastern europe","japan","budget travel","south america","cruises","honeymoons/getaways","air travel","africa","business travel","caribbean","national parks","hotels","by us locale","theme parks","bed & breakfasts","france","adventure travel","traveling with kids","italy","europe")
    val iab21 = List("iab21", "real estate", "apartments","buying/selling homes","architects")
    val iab22 = List("iab22", "shopping", "engines","contests & freebies","couponing","comparison")
    val iab23 = List("iab23", "religion", "spirituality", "atheism/agnosticism","islam","buddhism","alternative religions","judaism","catholicism","latter-day saints","pagan/wiccan","hinduism","christianity")
    val iab24 = List("iab24", "uncategorized")
    val iab25 = List("iab25", "non-standard content", "unmoderated ugc", "extreme graphic/explicit violence", "pornography", "profane content", "hate content", "under construction", "incentivized")
    val iab26 = List("iab26", "illegal content", "spyware/malware","warez","illegal content","copyright infringement")


    val first13Interests = interestsAsStr.map(value => {
      var tuple13 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      val interestValue = value.toString.toLowerCase

      tuple13 = tuple13.copy(
        _1 = exists(iab1, interestValue),
        _2 = exists(iab2, interestValue),
        _3 = exists(iab3, interestValue),
        _4 = exists(iab4, interestValue),
        _5 = exists(iab5, interestValue),
        _6 = exists(iab6, interestValue),
        _7 = exists(iab7, interestValue),
        _8 = exists(iab8, interestValue),
        _9 = exists(iab9, interestValue),
        _10 = exists(iab10, interestValue),
        _11 = exists(iab11, interestValue),
        _12 = exists(iab12, interestValue),
        _13 = exists(iab13, interestValue))
      tuple13
    })

    val last13Interests = interestsAsStr.map(value => {
      var tuple13 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      val interestValue = value.toString.toLowerCase

      tuple13 = tuple13.copy(
        _1 = exists(iab14, interestValue),
        _2 = exists(iab15, interestValue),
        _3 = exists(iab16, interestValue),
        _4 = exists(iab17, interestValue),
        _5 = exists(iab18, interestValue),
        _6 = exists(iab19, interestValue),
        _7 = exists(iab20, interestValue),
        _8 = exists(iab21, interestValue),
        _9 = exists(iab22, interestValue),
        _10 = exists(iab23, interestValue),
        _11 = exists(iab24, interestValue),
        _12 = exists(iab25, interestValue),
        _13 = exists(iab26, interestValue))
      tuple13
    })

    val l1 = first13Interests.toDF("i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10", "i11", "i12", "i13")
    val l2 = last13Interests.toDF("i14", "i15", "i16", "i17", "i18", "i19", "i20", "i21", "i22", "i23", "i24", "i25", "i26")

    val concatenatedValues = dataFrame.withColumn("id", monotonically_increasing_id())
      .join(l1.withColumn("id", monotonically_increasing_id()), Seq("id"))
      .join(l2.withColumn("id", monotonically_increasing_id()), Seq("id"))

    concatenatedValues
  }
}
