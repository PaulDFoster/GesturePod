/*
 * This is an autogenerated file. Modifications
 * might not be persistent.
 */
namespace protoNNParam {

#ifndef __TEST_PROTONN__
    /** Gamma for gaussian kernel */
    const PROGMEM float gamma = 0.001475;
    /** Low Dimensional Projection Matrix */
    const PROGMEM unsigned int featDim = 124;
    const PROGMEM unsigned int ldDim = 10;

    /**
    * Projectino Matrix (W)
    * d_cap x d flattened (dimension of 2D array)
    * ldDim x featDim
    */
    const PROGMEM  float ldProjectionMatrix[]  = {
-0.120428,6.312910,-0.091572,-0.467159,0.088151,0.505264,-0.907499,0.710649,-30.659822,5.237339,8.948300,7.228034,5.858997,-3.753675,-1.633528,-0.559220,-1.490238,-1.571604,-0.811170,2.022086,42.833904,-1.030105,-0.469014,0.634188,0.868470,2.012949,2.010990,-0.601670,-0.745723,0.800468,1.354370,-7.474811,1.379222,1.161115,0.185732,-3.486694,-4.158976,1.674214,1.726012,1.668787,-11.667713,35.137951,0.266846,0.391871,-1.858617,0.464775,-0.924804,-0.907662,0.166191,0.653278,-7.909085,-7.210186,-6.848476,-0.505882,-1.206203,0.607733,0.648744,-0.181736,2.735837,2.254147,0.991177,0.255269,-0.485418,-0.384402,-0.804315,-0.032248,-0.601439,0.264529,0.527223,-0.019397,-96.232422,18.993309,-11.666340,0.529153,0.381018,-16.500540,-95.333015,0.864215,-1.608112,1.006877,1.538684,-0.435438,0.064465,0.039569,-1.134989,-0.440374,0.749088,-1.435647,-0.701811,0.496914,-8.542283,-15.274717,-2.132968,0.388557,1.844810,0.507440,-3.687588,1.294432,-0.006192,0.467057,0.228064,-1.771613,0.851712,-0.808483,-0.320096,0.842467,-1.096298,-1.183931,-9.663909,-11.551405,1.333032,112.468880,-1.541630,-0.885563,-1.175542,6.161488,-2.392753,-13.502505,0.458998,1.366919,0.215229,-0.700507,-0.470716,-1.377286,
0.105931,5.366639,0.378937,-2.014976,-0.481790,0.313076,-0.234286,2.695826,-13.705183,0.204811,0.121534,-0.985557,0.290901,0.976816,0.006480,0.695024,1.616309,2.286654,0.144140,3.623003,-9.320165,1.277382,0.964170,0.091661,-1.340053,0.468869,0.032299,0.643458,0.441929,-1.422716,1.474006,11.354934,-0.830104,-0.840032,-1.116480,-0.473144,1.671396,-0.775585,-1.067842,2.324841,5.399617,-47.639332,-1.516802,-0.002933,0.991449,1.721712,-1.435642,-0.211522,0.571885,1.165627,-0.606729,0.890423,0.296978,-0.160754,-1.603081,1.464511,0.487229,0.126212,-4.492616,0.113750,1.553248,-0.544241,0.420080,0.099307,-0.242247,-1.532757,0.048166,1.944435,-0.666532,0.487375,-1.749839,-13.004144,10.279013,-0.304393,-0.866353,2.354033,6.099041,0.656952,-1.255832,1.275098,0.986299,-0.364605,2.100601,0.007772,1.233893,0.250467,1.386937,-0.459692,0.609350,-0.066671,-33.492569,-6.901078,14.587798,0.294084,0.336485,6.687129,-3.430701,0.090420,-0.525940,0.045920,0.049708,0.189197,0.273070,0.140968,-0.295116,0.327532,1.159930,1.886666,-33.962234,-35.259972,-34.904514,-113.298347,7.429510,-1.951099,0.120146,-9.163110,-8.832225,27.703474,0.016762,-0.089984,1.496355,1.120224,0.775651,0.023607,
-0.219264,6.983657,0.117641,1.383198,-1.716956,0.460828,0.493414,-0.470082,-20.463287,1.757656,3.346816,-0.094151,2.142562,0.331630,0.067735,0.933655,-0.190499,13.553869,17.048407,19.388746,25.687040,-0.938613,0.385775,-0.337857,1.499827,0.753549,-0.249378,-2.270966,0.209903,-0.878077,-0.753684,4.137068,0.794603,-0.492866,-0.991364,-3.315963,-0.195221,-2.335334,-2.410592,0.139322,9.568149,-44.366116,1.083858,-1.154704,-2.578444,2.048455,0.039231,-0.561757,0.203435,1.235712,-1.184919,-2.995161,-3.626076,0.568768,-0.738709,0.991098,1.390299,0.872270,-4.412601,-0.722830,0.902771,-0.879153,-0.734426,0.004437,0.237450,0.335458,-0.875460,0.010114,-0.466556,1.149773,-54.705563,-10.870398,8.238725,1.026808,0.204442,2.581394,-50.925968,-0.745752,-0.406785,-0.764232,0.339982,0.216245,1.037899,1.433353,-1.570962,-0.462699,0.644702,-1.033676,-0.611837,-1.807862,-32.492485,-4.456873,-1.293541,-0.358847,-0.919763,-3.925310,7.985608,-2.686697,0.577789,1.524449,1.315420,-0.693681,-1.319712,-1.250238,-0.917028,1.543086,0.485666,0.044408,-32.002380,-36.182640,-30.403904,-53.307587,9.989502,0.298364,0.779678,-0.395365,-8.430746,1.384248,-0.835457,-0.826944,-1.344692,1.557748,-0.393713,-2.011421,
0.327140,-6.612496,1.378521,-0.566137,-1.147807,0.086512,0.260738,0.402349,29.553490,-4.741694,-5.563974,-4.727021,-3.531765,0.754289,0.344309,-1.467344,1.088747,-5.889026,-8.352519,-11.025544,-35.970329,-0.060601,-1.169638,1.500821,-0.132086,-1.432020,-0.365074,-0.208275,-0.249792,0.881574,0.491155,1.986973,-2.007924,0.375824,0.572763,3.156185,2.229894,0.604111,-0.219841,-0.629595,-7.452147,-8.652502,0.004310,0.533345,-0.530391,-1.155143,1.369367,-0.732528,-1.013180,-1.301752,3.222881,5.062963,4.828242,-0.338518,0.193203,-0.055354,-0.766825,-0.953791,0.790870,-1.537925,-0.508604,0.153360,-0.134899,-2.163842,-1.327177,-0.439155,0.793681,-0.959357,0.073678,-1.221926,79.311363,-0.934876,2.579107,-0.245985,0.421012,4.350701,77.581161,0.123211,0.785423,0.340847,1.585509,0.532680,-0.161244,0.696949,0.653987,-0.752485,-0.355129,-1.150543,-0.275448,-0.488499,15.471833,19.518789,4.229360,-0.366726,-0.122646,0.900870,-2.979795,-2.657030,-0.552323,0.988018,1.191461,-1.055600,-0.426787,0.964594,-0.634477,0.925558,-1.479437,-1.533566,17.301353,22.598467,14.761803,-67.754707,-2.624886,-0.018175,0.854852,-5.016395,5.023068,11.474790,0.746717,0.066852,-0.408921,1.207418,-0.648990,-0.659955,
-0.114062,9.025163,-0.270173,0.003253,2.532475,0.929428,0.773176,-1.113084,-23.647562,1.886898,4.957888,1.818055,3.099113,-0.527152,-0.313794,0.367236,0.556842,7.661785,11.068473,15.397824,20.693426,1.106032,-0.343499,-0.151231,1.220793,0.351639,1.308028,0.527794,-0.343568,1.082464,0.659109,4.653205,1.147589,0.554275,0.036983,-3.056030,-0.281073,-0.980836,-1.647617,1.470340,1.946380,-49.324940,-0.280499,0.063817,-1.207780,0.644573,2.079102,0.357401,0.022326,1.790116,-1.885891,-1.600266,-2.390707,1.988782,0.314396,3.475809,3.124510,2.092278,-2.213193,0.883479,-0.762847,0.637725,0.195793,1.507779,-0.249194,-0.936432,0.646642,0.973324,0.603000,-2.872252,-57.843735,-12.338592,8.240769,-0.305350,-1.081817,-3.910516,-56.818546,1.098550,1.378402,0.672824,-1.179327,-1.303401,0.677479,1.176049,-1.224748,-0.339503,-0.070110,-2.077604,0.673727,0.106312,-33.880543,-6.351560,4.125898,-1.753300,-1.264887,-1.115625,3.221584,-0.826802,1.310148,0.114764,-0.984904,0.543581,-1.272477,0.193672,0.333353,-0.823717,-0.248765,0.963387,-34.175251,-36.285957,-34.469547,-63.489170,10.862041,-0.608470,1.108239,-3.089947,-8.070053,5.483911,0.462244,-0.562271,0.881204,0.037048,-1.163806,0.597103,
0.141326,-4.047873,-0.195249,1.688569,-0.548801,-0.130280,1.313096,0.391528,30.266298,-4.557644,-8.488004,-7.072189,-6.039003,2.562361,0.989933,-0.519723,0.922611,-0.928093,-1.654138,-5.460425,-40.630199,0.040029,0.423774,0.399203,0.089036,-1.387221,-0.351382,0.037356,1.314161,-0.392694,0.263936,5.659679,-2.409815,-1.160939,-0.684110,2.395717,3.937651,-0.618588,-1.161359,-1.408774,9.890558,-32.669319,-0.289281,-0.795642,-0.250855,1.899675,-0.140701,-0.848857,-0.859352,-1.412752,7.476936,6.893370,6.516946,0.682174,1.466226,0.088326,-0.125378,0.185078,-0.794628,-0.759831,0.895097,0.196527,1.320874,-0.710557,1.232558,-0.114097,1.777512,-1.095790,-1.013271,-0.368675,88.727905,-12.262701,10.487677,0.736661,1.094368,14.660791,88.544319,-1.859982,0.857615,-0.272909,0.492123,-0.584307,-0.282076,-0.673866,-0.141212,-0.300979,-0.190815,-0.360075,0.717374,1.397653,12.977411,21.044289,1.128574,-0.870236,-1.851019,-1.287293,1.776685,-1.115443,-0.222708,-0.191496,-0.262031,-1.275416,0.414183,1.050688,0.375215,-0.750610,0.398175,0.547912,14.819645,17.326880,3.732054,-98.851257,-0.417630,-0.282148,0.163321,-6.257425,2.200885,11.487678,-1.999058,1.778114,-0.861013,0.518915,0.545938,0.512329,
0.265043,-3.792696,0.378570,-1.239243,-0.108157,-0.077948,0.237459,0.210898,17.598215,0.330616,-1.107492,1.885441,-0.829323,0.179241,0.558810,-0.068642,1.748959,-17.126114,-22.311100,-24.519251,-27.708515,1.856461,-0.115224,-0.258720,0.254794,1.276040,-0.850501,-0.780212,1.878132,0.485535,-0.131549,-3.904866,-1.424042,-0.290618,0.280493,2.464816,-0.931964,2.204623,2.004382,0.156936,-14.225726,39.239601,-2.179456,-0.407377,-1.179251,-0.086961,-0.300190,0.146546,-0.713136,-0.350298,-0.630884,1.600396,2.028953,-1.369934,-0.420563,-1.188920,-2.109152,-1.813122,3.429490,0.396741,1.355238,-0.728478,1.588892,-1.197639,0.166724,1.134045,0.692092,0.145590,1.569430,0.102311,52.868183,12.718236,-5.373311,0.649094,1.323413,-4.103995,49.487766,-1.217334,0.827439,-0.167838,-0.129363,-1.283849,-2.031475,-0.225305,0.034827,-0.001126,-0.053402,2.044561,-0.288801,-0.141388,29.074406,3.684067,7.137221,-1.221024,-0.102254,7.012847,-11.216090,-0.671201,0.567692,0.969803,-0.053182,0.394145,0.567785,2.443797,0.174076,-1.537632,-0.020557,-0.933780,27.978065,32.030346,31.199476,49.595055,-9.633757,-1.036829,-0.676724,-2.451506,8.818542,2.314397,-1.056503,0.886156,-0.318442,-0.325919,-1.631215,0.036928,
-0.192623,7.619864,0.756289,-0.134844,-1.736483,-0.737559,1.355578,-0.631579,-32.699497,4.860651,9.486677,8.431115,6.688890,-3.194919,-1.079511,0.446967,-0.945849,-0.083913,2.251294,7.538856,46.019650,-0.293993,-0.298117,-1.183237,-0.695048,-0.216826,-0.000963,-1.635234,0.341576,0.311739,0.165621,-9.094791,0.819005,0.678446,-0.181648,-4.287437,-5.573961,0.445971,0.594422,0.415559,-11.132048,34.858566,1.309441,0.834576,-0.246169,-1.294894,2.407825,0.728683,0.471783,0.114494,-7.258471,-5.997788,-5.493708,0.645813,-0.020953,1.557746,1.693038,1.024808,3.597408,2.553558,-0.899772,-0.990949,0.338013,-0.326098,1.019504,1.397802,0.610948,3.204890,1.454417,-0.517472,-98.418083,18.788303,-11.934109,0.269314,0.344351,-16.546331,-97.389412,-1.050639,1.301010,-0.441977,-0.325789,0.654375,0.892695,0.613105,-1.348657,1.516399,0.189222,-0.829772,1.026523,0.691408,-8.336995,-19.493702,-7.309056,-1.214179,0.031085,-0.813430,-0.387782,-0.321810,-0.329974,1.291010,-0.553960,0.166156,0.351447,-1.540561,1.439798,-0.645458,-0.545672,0.841674,-7.013910,-10.708671,1.029739,113.683426,0.489101,0.488935,0.151206,7.553818,-0.612254,-12.849255,-0.175171,-1.556595,-0.675737,0.939125,-0.356867,-0.156154,
0.242679,-2.774970,0.181844,-1.423323,-1.438459,-0.482493,-0.157547,-0.170843,29.712000,-3.983499,-6.904822,-5.568051,-4.139461,1.139553,0.466902,-1.589033,0.857259,-2.438997,-2.326418,-5.938167,-37.092968,-0.133608,0.268069,-0.768026,-1.426998,1.203582,-0.754063,0.279870,0.694444,0.213938,-1.641142,2.827044,-3.005416,-0.438597,-0.363205,1.169342,3.233341,0.712313,-0.409386,-0.605022,-1.750735,-24.295818,-0.811291,-0.483263,1.327811,1.541229,0.959629,0.867911,-0.474912,-0.294595,3.835160,4.905777,4.555348,-0.792328,-0.210232,-0.447922,-0.939589,-1.372347,0.512824,-1.192725,0.313713,-0.600034,0.251883,-0.443080,1.335970,1.694254,0.550578,0.173278,1.244844,-0.169902,87.173988,-5.832897,6.373492,0.118025,0.685834,7.955166,88.483276,-1.147027,-2.357203,0.600023,-0.315798,0.460576,0.457574,-0.688610,0.396430,-1.087824,-0.500386,-0.413580,0.246739,1.050519,19.700634,24.700771,1.250834,0.125858,0.187746,-1.135669,0.596036,-2.185068,-0.745682,-0.572762,-1.660958,-1.940764,0.044830,1.314952,-1.367938,-0.537909,-0.945461,-1.706688,16.878319,24.899786,14.220498,-83.931343,-0.558818,-0.450648,0.507712,-6.254879,5.227206,9.375451,-0.040035,-1.919333,-1.539798,-0.870680,-0.403462,0.469217,
0.282583,-5.074983,-0.202694,0.227069,0.046692,0.313451,1.247704,-0.580298,29.808678,-4.418484,-6.746529,-5.231002,-4.239182,2.325642,1.382153,-0.560917,1.775577,-4.731621,-6.951862,-10.883512,-37.085781,1.448650,1.363432,0.566668,0.380436,-0.348883,-0.266244,-1.715130,0.710264,0.200584,-1.688439,2.640417,-1.898681,0.303978,0.578332,3.703758,3.078401,0.281128,-0.403810,-1.056798,1.759325,-10.967145,-1.889377,-0.596838,0.684530,-0.815188,-1.188083,-2.158988,0.199005,1.254588,5.098611,5.538774,5.264738,-0.348845,0.426646,-0.554389,-1.119486,-1.028288,0.226001,-2.104893,-0.304337,0.071012,0.949144,0.016702,-0.362187,-1.078984,-0.522814,-1.527970,1.105529,0.686115,90.775780,-1.618301,2.225592,-0.282489,0.224646,9.012034,91.753548,-0.293667,1.180816,-1.373958,1.993506,1.721359,1.098374,1.462051,-0.412243,-1.042070,1.027945,-0.221698,0.234922,1.024797,17.645638,19.544710,4.103294,-0.171551,-0.369218,-0.012664,-2.433766,-0.076112,-1.502506,1.663728,0.929660,-0.994226,-0.021396,0.271861,-1.940913,0.462032,-1.026989,-0.545698,19.375187,23.832947,14.049661,-73.465569,-3.061005,-0.390110,0.209976,-5.322320,4.172835,10.442546,-2.292690,-1.463713,-0.284779,-0.341832,1.174193,-1.442076

	};
    
    /**
     * Prototypes (B)
     * m x d_cap flattened (dimension of 2D array)
     * numPrototypes x d_cap
     */
    const PROGMEM float prototypeMatrix[] = {
-50.902134,113.315895,43.656139,50.328560,119.318954,56.352940,-5.992833,-61.005711,67.065308,48.766655,
51.308426,71.510246,94.964897,-74.396118,85.444641,-69.993645,-63.344440,46.522831,-96.330208,-80.571205,
60.078041,-91.695213,-70.080002,-26.441086,-89.331467,-59.389202,51.797829,62.953526,-57.806816,-27.705185,
-56.204021,72.063820,-28.303791,52.058121,-0.558406,56.459900,44.247250,-66.045143,55.904984,61.758114,
79.983330,55.769741,80.941193,-74.693268,83.389412,-91.253937,-54.040581,73.230629,-98.500755,-89.775795,
-50.179886,126.525597,144.709122,-50.568848,119.096687,13.143586,-131.136520,-52.472004,-29.842690,-43.154705,
80.756897,47.340832,96.333694,-89.743973,88.245819,-90.669113,-79.563026,85.123352,-98.702278,-106.644936,
58.414955,-85.749611,-111.251846,44.192841,-90.451057,-20.715734,111.582191,52.537312,36.497078,27.602745,
-52.596073,97.327530,-10.314114,49.546291,51.787777,51.953484,36.496361,-63.324261,47.688095,56.340889,
145.059097,-53.919220,24.751350,-79.394714,43.413059,-137.577316,-28.490414,152.643204,-93.788544,-102.304901,
-100.079559,117.894180,136.513168,38.915802,122.186508,94.961342,-127.732712,-101.213959,81.258995,55.674759,
-67.816727,18.796171,-53.285809,57.236423,-55.849144,70.695953,51.667301,-73.385201,63.498203,70.561928,
-7.381551,-105.945923,-86.725609,33.406193,-112.475601,23.888557,63.215565,-7.184753,32.745846,44.772312,
113.825958,37.469513,83.938179,-84.210228,95.006523,-115.934952,-70.036308,117.530846,-97.866035,-108.394440,
29.861153,-106.220261,-122.601311,71.752747,-109.912910,14.134721,112.661644,25.592598,72.687950,62.390160,
-75.860184,132.770813,135.646698,13.790237,128.812531,59.251198,-104.850960,-82.828545,36.216400,24.182909,
-27.838108,-109.349365,-85.424019,36.331661,-108.685669,41.805977,57.698807,-25.825672,47.821823,45.398178,
-41.316639,-106.258080,-58.699947,19.738863,-95.870827,46.077034,24.811695,-40.451092,42.323082,35.117992,
28.733032,-123.004448,-44.792850,-42.081875,-95.281502,-34.501167,9.179164,36.971752,-47.584389,-27.767815,
-84.600739,61.135868,15.376832,53.100052,32.192234,81.213875,-9.523223,-88.465462,72.730347,64.488754

	};
    /** Number of prototypes (m) */
    const PROGMEM unsigned int numPrototypes = 20;
    
    /**
     * Prototype Lables (Z)
     * m x L (dimension of 2D array)
     * numLabels x numPrototypes
     */
    const PROGMEM float prototypeLabelMatrix[] = {
-22.633684,0.194643,-21.843292,11.679637,-2.366389,4.136011,-20.766504,-5.796654,-23.563324,-4.111383,
-17.447638,-14.852242,-17.129766,-6.753134,16.376240,2.879771,-18.158895,2.712160,-17.813553,5.385801,
-18.241713,9.372271,-18.150194,-22.921803,-15.271763,-1.854368,-17.582159,3.990956,-17.645351,3.882795,
-21.614965,6.963005,-20.815619,5.582535,-6.150159,5.875327,-19.273121,-2.745955,-19.115860,-4.118541,
-16.760374,-12.486115,-16.261871,-9.830523,16.352581,2.009561,-15.373030,3.200251,-16.670965,7.547641,
-22.737267,-14.914575,-21.908184,12.853152,18.262182,5.626729,-22.299683,2.459348,-20.836899,-1.723932,
-17.030094,-14.764572,-16.061199,-8.004401,16.521799,-0.430349,-15.132583,3.518841,-16.544256,7.242023,
-18.118301,14.241696,-18.251112,-11.618325,-19.053278,-0.997657,-18.980188,-1.853548,-19.000614,2.364790,
-22.187346,3.834148,-21.594273,5.216038,-3.436614,5.256880,-20.518057,-5.047806,-19.520031,-5.166605,
-14.205515,-2.942018,-15.196647,-20.801764,-2.832306,-3.587212,-14.283656,3.761874,-15.418726,11.595766,
-21.607796,-5.715021,-22.664677,27.760252,-0.989436,4.425398,-23.542625,-2.029813,-26.760532,-9.316497,
-17.915415,8.342625,-21.552383,3.061402,-11.590606,1.344570,-18.980669,-3.770008,-18.879583,-9.023754,
-18.483913,14.392115,-19.227381,-9.829086,-20.398106,-3.256670,-19.390360,1.622076,-17.934734,-3.768407,
-17.573063,-12.634370,-15.062524,-9.593164,14.946855,-1.592251,-16.788607,2.467065,-14.816257,8.604453,
-16.324137,18.761005,-17.670805,-7.426442,-22.987509,-3.706447,-19.461090,-2.083843,-17.059742,-0.206424,
-23.834408,-7.840858,-24.472425,19.246508,7.640802,6.598890,-23.306942,-1.081515,-22.436920,-4.834180,
-19.294252,12.226434,-18.554007,-5.051319,-21.521305,-2.056572,-19.530111,1.558624,-17.281639,-4.019403,
-18.831078,8.787330,-18.428547,-1.445534,-18.465054,-2.441643,-18.363220,3.115513,-19.732264,-5.278980,
-19.274240,6.597408,-17.352192,-18.970690,-16.201756,-4.821322,-17.755320,6.997601,-17.772039,1.839633,
-21.668528,2.885352,-19.602594,10.810181,-5.375288,2.813062,-21.625771,-2.018648,-22.633152,-6.518638

};
    /** Number of output labels, (L). */
    const PROGMEM unsigned int numLabels = 10; // 0,1,2,3,4,5
    
#else
    const PROGMEM float = 1.0;
    const PROGMEM unsigned int featDim = 10;
    const PROGMEM unsigned int ldDim = 5;
    // Row Major (X.x)
    const PROGMEM  float ldProjectionMatrix[]  = {
        0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,
        1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
        2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,
        3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,
        4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,
    };
    // Column Major
    const PROGMEM unsigned int numPrototypes = 3;
    const PROGMEM float prototypeMatrix[] = {
        -1.0,-0.5,0.0,0.5,1.0,
        -2.0,-1.0,0.0,1.0,2.0,
        -7.51,-7.51,-7.51,-7.51,-7.51,
    };
    // column major
    const PROGMEM unsigned int numLabels = 4;
    const PROGMEM float prototypeLabelMatrix[] = {
        0.96,0.01,0.01,0.02,
        0.02,0.94,0.02,0.02,
        0.10,0.15,0.55,0.20,
    };
#endif
};
    