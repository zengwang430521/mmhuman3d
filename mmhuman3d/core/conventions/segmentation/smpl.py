"""Raw index information can be found from smpl-wiki website:

https://meshcapade.wiki/SMPL#mesh-templates--samples
"""
SMPL_SEGMENTATION_DICT = {
    'rightHand':
    [[5442, 5487], [5492, 5497], [5502, 5527], [5530, 5562], [5569], [5571],
     [5574, 5583], [5588, 5589], [5592, 5605], [5610, 5614], [5621, 5622],
     [5625], [5631, 5641], [5643, 5646], [5649, 5650], [5652, 5664], [5667],
     [5670, 5675], [5682, 5690], [5692], [5695], [5697, 5701], [5707, 5721],
     [5723, 5732], [5735, 5740], [5745, 5746], [5748, 5752], [6056, 6057],
     [6066, 6067], [6158, 6239]],
    'rightUpLeg': [[4320, 4321], [4323, 4324], [4333, 4340], [4356, 4367],
                   [4383, 4401], [4419, 4422], [4430, 4532], [4623, 4634],
                   [4645, 4660], [4670, 4673], [4704, 4713], [4745, 4746],
                   [4757, 4760], [4801, 4802], [4829], [4834, 4841],
                   [4924, 4926], [4928, 4936], [4948, 4952], [4970, 4973],
                   [4983, 4993], [5004, 5005], [6546, 6549], [6552, 6556],
                   [6873], [6877]],
    'leftArm': [[626, 629], [634, 635], [680, 681], [716, 719], [769, 780],
                [784, 793], [1231, 1234], [1258, 1261], [1271], [1281, 1282],
                [1310, 1311], [1314, 1315], [1340, 1343], [1355, 1358],
                [1376, 1400], [1402, 1403], [1405, 1416], [1428, 1433],
                [1438, 1445], [1502], [1505, 1510], [1538],
                [1541, 1543], [1545], [1619, 1622], [1631, 1642], [1645, 1656],
                [1658, 1659], [1661, 1662], [1664], [1666, 1684], [1696, 1698],
                [1703, 1720], [1725], [1731, 1735], [1737], [1739, 1740],
                [1745, 1749], [1751], [1761], [1830, 1831], [1844, 1846],
                [1850, 1851], [1854, 1855], [1858], [1860], [1865, 1867],
                [1869, 1871], [1874, 1878], [1882, 1883], [1888, 1889], [1892],
                [1900, 1904], [1909], [2819, 2822], [2895, 2903], [2945, 2946],
                [2974, 2996], [3002], [3013]],
    'leftLeg': [[995], [998, 999], [1002], [1004, 1005], [1008], [1010],
                [1012], [1015, 1016], [1018, 1019], [1043, 1044], [1047, 1136],
                [1148, 1158], [1175, 1183], [1369, 1375], [1464, 1474],
                [1522, 1532], [3174, 3210], [3319, 3335], [3432, 3436], [3469],
                [3472, 3474]],
    'leftToeBase': [[3211, 3318], [3336, 3337], [3340], [3342], [3344], [3346],
                    [3348], [3350], [3352], [3354], [3357, 3358], [3360],
                    [3362]],
    'leftFoot': [[3327, 3469]],
    'spine1':
    [[598, 601], [610, 621], [642], [645, 647], [652, 653], [658, 661],
     [668, 671], [684, 692], [722, 725], [736], [750, 751], [761], [764],
     [766, 767], [794, 795], [891, 894], [925, 929], [940, 943], [1190, 1197],
     [1200, 1202], [1212], [1236], [1252, 1255], [1268, 1270], [1329, 1330],
     [1348, 1349], [1351], [1420, 1421], [1423, 1426], [1436, 1437],
     [1756, 1758], [2839, 2851], [2870, 2871], [2883], [2906], [2908], [3014],
     [3017], [3025], [3030], [3033, 3034], [3037], [3039, 3044], [3076, 3077],
     [3079], [3480], [3505], [3511], [4086, 4089], [4098, 4109], [4130, 4131],
     [4134, 4135], [4140, 4141], [4146, 4149], [4156, 4159], [4172, 4180],
     [4210, 4213], [4225], [4239, 4240], [4249, 4250], [4255, 4256],
     [4282, 4283], [4377, 4380], [4411, 4415], [4426, 4429], [4676, 4683],
     [4686, 4688], [4695], [4719], [4735, 4737], [4740], [4751, 4753],
     [4824, 4825], [4828], [4893, 4895], [4897, 4899], [4908, 4909],
     [5223, 5225], [6300, 6312], [6331, 6332], [6342], [6366, 6367], [6475],
     [6477, 6478], [6481, 6482], [6485], [6487, 6491], [6878]],
    'spine2': [[570, 573], [584, 597], [602, 609], [622, 625], [638, 641],
               [643, 644], [648, 651], [666, 667], [672, 675], [680, 683],
               [693, 704], [713, 717], [726, 733], [735],
               [737, 749], [752, 760], [762, 763], [803, 806], [811, 814],
               [817, 821], [824, 828], [895, 896], [930, 931], [1198, 1199],
               [1213, 1220], [1235], [1237], [1256, 1257], [1271, 1273],
               [1279, 1280], [1283, 1309], [1312, 1313], [1319, 1320],
               [1346, 1347], [1350], [1352], [1401], [1417, 1419], [1422],
               [1427], [1434, 1435], [1503, 1504], [1536, 1537], [1544, 1545],
               [1753, 1755], [1759, 1763], [1808, 1811], [1816, 1820],
               [1834, 1839], [1868], [1879, 1880], [2812, 2813], [2852, 2869],
               [2872], [2875, 2878], [2881, 2882], [2884, 2886], [2904, 2905],
               [2907], [2931, 2937], [2941], [2950, 2973], [2997, 2998],
               [3006, 3007], [3012], [3015], [3026, 3029], [3031, 3032],
               [3035, 3036], [3038], [3059, 3067], [3073, 3075], [3078],
               [3168, 3169], [3171], [3470, 3471], [3482, 3483], [3495, 3498],
               [3506], [3508], [4058, 4061], [4072, 4085], [4090, 4097],
               [4110, 4113], [4126, 4129], [4132, 4133], [4136, 4139],
               [4154, 4155], [4160, 4163], [4168, 4171], [4181, 4192],
               [4201, 4204], [4207], [4214, 4221], [4223, 4224], [4226, 4238],
               [4241, 4248], [4251, 4252], [4291, 4294], [4299, 4302],
               [4305, 4309], [4312, 4315], [4381, 4382], [4416, 4417],
               [4684, 4685], [4696, 4703], [4718], [4720], [4738, 4739],
               [4754, 4756], [4761, 4762], [4765, 4789], [4792, 4793],
               [4799, 4800], [4822, 4823], [4826, 4827], [4874], [4890, 4892],
               [4896], [4900], [4907], [4910], [4975, 4976], [5007, 5008],
               [5013, 5014], [5222], [5226, 5230], [5269, 5272], [5277, 5281],
               [5295, 5300], [5329], [5340, 5341], [6273, 6274], [6313, 6330],
               [6333], [6336, 6337], [6340, 6341], [6343, 6345], [6363, 6365],
               [6390, 6396], [6398], [6409, 6432], [6456, 6457], [6465, 6466],
               [6476], [6479, 6480], [6483, 6484], [6486], [6496,
                                                            6503], [6879]],
    'leftShoulder': [[591], [604, 606], [609], [634, 637], [674], [706, 713],
                     [715], [717], [730], [733, 735], [781, 783], [1238, 1245],
                     [1290, 1291], [1294], [1316, 1318], [1401, 1404], [1509],
                     [1535], [1545], [1808], [1810, 1815], [1818, 1819],
                     [1821, 1833], [1837], [1840, 1859], [1861, 1864],
                     [1872, 1873], [1880, 1881], [1884, 1887], [1890, 1891],
                     [1893, 1899], [2879, 2881], [2886, 2894], [2903],
                     [2938, 2949], [2965], [2967], [2969], [2999, 3005],
                     [3008, 3011]],
    'rightShoulder': [[4077], [4091, 4092], [4094, 4095], [4122, 4125], [4162],
                      [4194, 4201], [4203], [4207], [4218, 4219], [4222, 4223],
                      [4269, 4271], [4721, 4728], [4773, 4774], [4778],
                      [4796, 4798], [4874, 4877], [4982], [5006], [5014],
                      [5269], [5271, 5276], [5279], [5281, 5294], [5298],
                      [5301, 5320], [5322, 5325], [5333, 5334], [5341, 5342],
                      [5345, 5348], [5351, 5352], [5354, 5360], [6338, 6340],
                      [6345, 6353], [6362], [6397, 6408], [6424, 6425], [6428],
                      [6458, 6464], [6467, 6470]],
    'rightFoot': [[6727, 6869]],
    'head': [[0, 149], [154, 173], [176, 205], [220, 221], [225, 255],
             [258, 283], [286, 295], [303, 304], [306, 307], [310, 332],
             [335, 422], [427, 439], [442, 450], [454, 459], [461, 569],
             [574, 583], [1764, 1766], [1770, 1778], [1905, 1908],
             [2779, 2811], [2814, 2818], [3045, 3048], [3051, 3056], [3058],
             [3069, 3072], [3161, 3163], [3165, 3167], [3485, 3494], [3499],
             [3512, 3661], [3666, 3685], [3688, 3717], [3732, 3733],
             [3737, 3767], [3770, 3795], [3798, 3807], [3815, 3816],
             [3819, 3838], [3841, 3917], [3922, 3933], [3936, 3941],
             [3945, 4057], [4062, 4071], [5231, 5233], [5235, 5243],
             [5366, 5369], [6240, 6272], [6275, 6279], [6492, 6495],
             [6880, 6889]],
    'rightArm': [[4114, 4117], [4122], [4125], [4168], [4171], [4204, 4207],
                 [4257, 4268], [4272, 4281], [4714, 4717], [4741,
                                                            4744], [4756],
                 [4763, 4764], [4790, 4791], [4794, 4795], [4816, 4819],
                 [4830, 4833], [4849, 4873], [4876, 4889], [4901, 4906],
                 [4911, 4918], [4974], [4977, 4982], [5009, 5012], [5014],
                 [5088, 5091], [5100, 5111], [5114, 5125], [5128, 5131],
                 [5134, 5153], [5165, 5167],
                 [5172, 5189], [5194], [5200, 5204], [5206], [5208, 5209],
                 [5214, 5218], [5220], [5229], [5292, 5293], [5303], [5306],
                 [5309], [5311], [5314, 5315], [5318, 5319], [5321],
                 [5326, 5328], [5330, 5332], [5335, 5339], [5343, 5344],
                 [5349, 5350], [5353], [5361, 5365], [5370], [6280, 6283],
                 [6354, 6362], [6404, 6405], [6433, 6455], [6461], [6471]],
    'leftHandIndex1': [[2027, 2030], [2037, 2040], [2057], [2067, 2068],
                       [2123, 2130], [2132], [2145, 2146], [2152, 2154],
                       [2156, 2169], [2177, 2179], [2181], [2186, 2187],
                       [2190, 2191], [2204, 2205], [2215, 2220], [2232, 2233],
                       [2245, 2247], [2258, 2259], [2261, 2263], [2269, 2270],
                       [2272, 2274], [2276, 2277], [2280, 2283], [2291, 2594],
                       [2596, 2597], [2599, 2604], [2606, 2607], [2609, 2696]],
    'rightLeg': [[4481, 4482], [4485, 4486], [4491, 4493], [4495], [4498],
                 [4500, 4501], [4505, 4506], [4529], [4532, 4622],
                 [4634, 4644], [4661, 4669], [4842, 4848], [4937, 4947],
                 [4993, 5003], [6574, 6610], [6719, 6735], [6832, 6836],
                 [6869, 6872]],
    'rightHandIndex1': [[5488, 5491], [5498, 5501], [5518], [5528, 5529],
                        [5584, 5592], [5606, 5607], [5613], [5615, 5630],
                        [5638, 5640], [5642], [5647, 5648], [5650, 5651],
                        [5665, 5666], [5676, 5681], [5693, 5694], [5706, 5708],
                        [5719], [5721, 5724], [5730, 5731], [5733, 5735],
                        [5737, 5738], [5741, 5744], [5752, 6055], [6058, 6065],
                        [6068, 6157]],
    'leftForeArm': [[1546, 1618], [1620, 1621], [1623, 1630], [1643, 1644],
                    [1646, 1647], [1650, 1651], [1654, 1655], [1657, 1666],
                    [1685, 1695], [1699, 1702], [1721, 1730], [1732], [1736],
                    [1738], [1741, 1744], [1750], [1752], [1900], [1909, 1980],
                    [2019], [2059, 2060], [2073], [2089], [2098, 2112],
                    [2147, 2148], [2206, 2209], [2228], [2230], [2234, 2235],
                    [2241, 2244], [2279], [2286], [2873, 2874]],
    'rightForeArm': [[5015, 5087], [5090, 5099], [5112, 5113], [5116, 5117],
                     [5120, 5121], [5124, 5135], [5154, 5164], [5168, 5171],
                     [5190, 5199], [5202],
                     [5205], [5207], [5210, 5213], [5219], [5221], [5361],
                     [5370, 5441], [5480], [5520, 5521], [5534], [5550],
                     [5559, 5573], [5608, 5609], [5667, 5670], [5689], [5691],
                     [5695, 5696], [5702, 5705], [5740], [5747], [6334, 6335]],
    'neck': [[148], [150, 153], [172], [174, 175], [201, 202], [204, 219],
             [222, 225], [256, 257], [284, 285], [295, 309], [333, 334],
             [423, 426], [440, 441], [451, 453], [460, 461], [571, 572],
             [824, 829], [1279, 1280], [1312, 1313], [1319, 1320], [1331],
             [3049, 3050], [3057, 3059], [3068], [3164], [3661, 3665],
             [3685, 3687], [3714, 3731], [3734, 3737], [3768, 3769],
             [3796, 3797], [3807, 3819], [3839, 3840], [3918, 3921],
             [3934, 3935], [3942, 3944], [3950], [4060, 4061], [4312, 4315],
             [4761, 4762], [4792, 4793], [4799, 4800], [4807]],
    'rightToeBase': [[6611, 6718], [6736], [6739], [6741], [6743], [6745],
                     [6747], [6749, 6750], [6752], [6754], [6757, 6758],
                     [6760], [6762]],
    'spine': [[616, 617], [630, 633], [654, 657], [662, 665], [720, 721],
              [765, 768], [796, 799], [889, 890], [916, 919], [921, 926],
              [1188, 1189], [1211, 1212], [1248, 1251], [1264, 1267],
              [1323, 1328], [1332, 1336], [1344, 1345], [1481, 1496], [1767],
              [2823, 2845], [2847, 2848], [2851], [3016, 3020], [3023, 3024],
              [3124], [3173], [3476, 3478], [3480], [3500,
                                                     3502], [3504], [3509],
              [3511], [4103, 4104], [4118, 4121], [4142, 4145], [4150, 4153],
              [4208, 4209], [4253, 4256], [4284, 4287], [4375, 4376],
              [4402, 4403], [4405, 4412], [4674, 4675], [4694, 4695],
              [4731, 4734], [4747, 4750], [4803, 4806], [4808, 4812],
              [4820, 4821], [4953, 4968], [5234], [6284, 6306], [6308, 6309],
              [6312], [6472, 6474], [6545], [6874, 6876], [6878]],
    'leftUpLeg': [[833, 834], [838, 839], [847, 854], [870, 881], [897, 915],
                  [933, 936], [944, 1046], [1137, 1148], [1159, 1174],
                  [1184, 1187], [1221, 1230], [1262, 1263], [1274, 1277],
                  [1321, 1322], [1354], [1359, 1362], [1365,
                                                       1368], [1451, 1453],
                  [1455, 1463], [1475], [1477, 1480], [1498, 1501],
                  [1511, 1514], [1516, 1522], [1533, 1534], [3125, 3128],
                  [3131, 3135], [3475], [3479]],
    'leftHand': [[1981, 2026], [2031, 2036],
                 [2041, 2066], [2069, 2101], [2107], [2111], [2113, 2122],
                 [2127], [2130, 2144], [2149, 2152], [2155], [2160],
                 [2163, 2164], [2170, 2180], [2182, 2185], [2188, 2189],
                 [2191, 2203], [2207], [2209, 2214], [2221, 2229], [2231],
                 [2234], [2236, 2240], [2246, 2260], [2262,
                                                      2271], [2274, 2279],
                 [2284, 2285], [2287, 2290], [2293], [2595], [2598], [2605],
                 [2608], [2697, 2778]],
    'hips': [[631, 632], [654], [657], [662], [665], [676, 679], [705], [720],
             [796], [799, 802], [807, 810], [815, 816], [822, 823], [830, 846],
             [855, 869], [871], [878], [881, 890], [912], [915, 920], [932],
             [937, 939], [1163], [1166], [1203, 1210], [1246, 1247],
             [1262, 1263], [1276, 1278], [1321], [1336, 1339], [1353, 1354],
             [1361, 1364], [1446, 1450], [1454], [1476], [1497], [1511],
             [1513, 1515], [1533, 1534], [1539, 1540], [1768, 1769],
             [1779, 1807], [2909, 2930], [3018, 3019], [3021, 3022],
             [3080, 3124], [3128, 3130], [3136, 3160], [3170], [3172], [3481],
             [3484], [3500], [3502, 3503], [3507], [3510], [4120, 4121],
             [4142, 4143], [4150, 4151], [4164, 4167], [4193], [4208],
             [4284, 4285], [4288, 4290], [4295, 4298], [4303, 4304],
             [4310, 4311], [4316, 4332], [4341, 4356], [4364, 4365],
             [4368, 4376], [4398, 4399], [4402, 4406], [4418], [4423, 4425],
             [4649, 4650], [4689, 4693], [4729, 4730], [4745, 4746],
             [4759, 4760], [4801], [4812, 4815], [4829], [4836, 4837],
             [4919, 4923], [4927], [4969], [4983, 4984], [4986], [5004, 5005],
             [5244, 5268], [6368, 6389], [6473, 6474], [6504, 6545],
             [6549, 6551], [6557, 6573]]
}


def smpl_part_segmentation(key):
    part_segmentation = []
    raw_segmentation = SMPL_SEGMENTATION_DICT[key]
    for continous in raw_segmentation:
        if len(continous) == 2:
            part_segmentation.extend(
                list(range(continous[0], continous[1] + 1)))
        elif len(continous) == 1:
            part_segmentation.extend(continous)
    return part_segmentation
