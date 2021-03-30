import pandas as pd





# Read Education File
def readEducation():
	education = pd.read_csv('education.csv')
	education = education.iloc[1:, 2:].rename(columns = {'village ': 'village'})
	education.iloc[:, 2:] = education.iloc[:, 2:].apply(pd.to_numeric)
	education = education.sort_values(by = ['site_id', 'village']).reset_index(drop = True)

	##
	# edu_doctor_graduated_m, edu_doctor_graduated_f
	##
	education['PhD'] = education.iloc[:, 3:5].sum(axis = 1)

	##
	# edu_doctor_ungraduated_m, edu_doctor_ungraduated_f, edu_master_graduated_m, edu_master_graduated_f
	##
	education['Master'] = education.iloc[:, 5:9].sum(axis = 1)

	##
	# edu_master_ungraduated_m, edu_master_ungraduated_f, edu_university_graduated_m, edu_university_graduated_f,
	# edu_juniorcollege_2ys_graduated_m, edu_juniorcollege_2ys_graduated_f, edu_juniorcollege_5ys_final2y_graduated_m,
	# edu_juniorcollege_5ys_final2y_graduated_f
	##
	education['Bachelor'] = education.iloc[:, [9, 10, 11, 12, 15, 16, 19, 20]].sum(axis = 1)

	##
	# edu_university_ungraduated_m, edu_university_ungraduated_f, edu_juniorcollege_2ys_ungraduated_m, edu_juniorcollege_2ys_ungraduated_f,
	# edu_juniorcollege_5ys_final2y_ungraduated_m, edu_juniorcollege_5ys_final2y_ungraduated_f, edu_senior_graduated_m, edu_senior_graduated_f,
	# edu_seniorvocational_graduated_m, edu_seniorvocational_graduated_f, edu_juniorcollege_5ys_first3y_ungraduated_m,
	# edu_juniorcollege_5ys_first3y_ungraduated_f
	##
	education['Senior'] = education.iloc[:, [13, 14, 17, 18, 21, 22, 23, 24, 27, 28, 31, 32]].sum(axis = 1)

	##
	# edu_senior_ungraduated_m, edu_senior_ungraduated_f, edu_seniorvocational_ungraduated_m, edu_seniorvocational_ungraduated_f,
	# edu_junior_graduated_m, edu_junior_graduated_f, edu_juniorvocational_graduated_m, edu_juniorvocational_graduated_f
	##
	education['Junior'] = education.iloc[:, [25, 26, 29, 30, 33, 34, 37, 38]].sum(axis = 1)

	##
	# edu_junior_ungraduated_m, edu_junior_ungraduated_f, edu_juniorvocational_ungraduated_m, edu_juniorvocational_ungraduated_f,
	# edu_primary_graduated_m, edu_primary_graduated_f
	##
	education['Elementary'] = education.iloc[:, [35, 36, 39, 40, 41, 42]].sum(axis = 1)

	##
	# edu_primary_ungraduated_m, edu_primary_ungraduated_f, edu_selftaughtl_m, edu_selftaughtl_f, edu_illiterate_m, edu_illiterate_f
	##
	education['illiterate'] = education.iloc[:, 43:49].sum(axis = 1)

	df1 = (education['PhD'].mul(7)).add(education['Master'].mul(6))
	df2 = (education['Bachelor'].mul(5)).add(education['Senior'].mul(4))
	df3 = ((education['Junior'].mul(3)).add(education['Elementary'].mul(2))).add(education['illiterate'].mul(1))
	df4 = (df1.add(df2)).add(df3)
	df6 = education['PhD'].add(education['Master'])
	df7 = education['Bachelor'].add(education['Senior'])
	df8 = (education['Junior'].add(education['Elementary'])).add(education['illiterate'])
	df9 = (df6.add(df7)).add(df8)
	education['EducationWeight'] = df4.divide(df9)

	education = education.drop(education.iloc[:, 3:56], axis = 1)

	return education




# Read Merriage File
def readMerriage():
	merriage = pd.read_csv('merriage.csv', low_memory = False).drop(0)
	merriage = merriage.drop(merriage.columns[[0, 3, 4, 22, 23, 41, 42, 60, 61, 79, 80, 98, 99, 107, 108, 126, 127]], axis = 1).sort_values(by = ['site_id', 'village']).reset_index(drop = True)
	merriage.iloc[:, 2:] = merriage.iloc[:, 2:].apply(pd.to_numeric)

	merriage['NoMerriage'] = merriage.iloc[:, 2:36].sum(axis = 1)

	merriage['Merriage'] = merriage.iloc[:, 36:].sum(axis = 1)

	merriage['Merriage/NoMerriage'] = merriage['Merriage'].divide(merriage['NoMerriage'])

	merriage = merriage.drop(merriage.iloc[:, 2:140], axis = 1)

	return merriage





# Read Living Situation File
def readLivingsituation():
	livingsituation = pd.read_csv('livingsituation.csv').iloc[1:, [1, 2, 3, 5]].sort_values(by = ['site_id', 'village']).reset_index(drop = True)
	livingsituation.iloc[:, 2:] = livingsituation.iloc[:, 2:].apply(pd.to_numeric)
	livingsituation['ordinary/single'] = livingsituation['household_ordinary_total'].divide(livingsituation['household_single_total'])

	livingsituation = livingsituation.drop(livingsituation.iloc[:, [2, 3]], axis = 1)

	return livingsituation





# Read Family File
def readFamily():
	family = pd.read_csv('family.csv').iloc[1:, 1:13].sort_values(by = ['site_id', 'village']).reset_index(drop = True)
	family.iloc[:, 2:] = family.iloc[:, 2:].apply(pd.to_numeric)

	family['One_to_Four'] = family.iloc[:, 2:6].sum(axis = 1)

	family['Five_to_Nine'] = family.iloc[:, 6:11].sum(axis = 1)

	family['Above_Ten'] = family.iloc[:, 11]

	family['FamilyWeight'] = (((family['One_to_Four'].mul(3)).add(family['Five_to_Nine'].mul(7))).add(family['Above_Ten'].mul(10))).divide((family['One_to_Four'].add(family['Five_to_Nine'])).add(family['Above_Ten']))

	family = family.drop(family.iloc[:, 2:15], axis = 1)

	return family





# Read Population File
def readPopulation():
	population = pd.read_csv('population.csv', low_memory = False).drop(0).sort_values(by = ['site_id', 'village']).reset_index(drop = True)
	population = pd.concat([population.iloc[:, [2, 3, 6, 7]], population.iloc[:, 44:]], axis = 1)
	population.iloc[:, 2:] = population.iloc[:, 2:].apply(pd.to_numeric)

	population['18-38'] = population.iloc[:, 4:46].sum(axis = 1)

	population['39-59'] = population.iloc[:, 46:88].sum(axis = 1)

	population['60-80'] = population.iloc[:, 88:130].sum(axis = 1)

	population['81-100'] = population.iloc[:, 130:170].sum(axis = 1)

	population['Total_Ratio'] = population.iloc[:, 2].divide(population.iloc[:, 3])

	df1 = (population['18-38'].mul(1)).add(population['39-59'].mul(2))
	df2 = (population['60-80'].mul(3)).add(population['81-100'].mul(4))
	df3 = (population['18-38'].add(population['60-80'])).add(population['39-59'].add(population['81-100']))
	population['18-59/60-100'] = (df1.add(df2)).divide(df3)

	population = population.drop(population.iloc[:, 2:170], axis = 1)

	return population





# Read Income File
def readIncome():
	filename = ['Changhua.csv', 'ChiayiCity.csv', 'ChiayiCounty.csv', 'HsinchuCity.csv', 'HsinchuCounty.csv', 'Hualien.csv', 'Kaohsiung.csv', 'Keelung.csv', 'Kinmen.csv', 'Lianjiang.csv', 'Miaoli.csv', 'Nantou.csv', 'NewTaipei.csv', 'Penghu.csv', 'Pingtung.csv', 'Taichung.csv', 'Tainan.csv', 'Taipei.csv', 'Taitung.csv', 'Taoyuan.csv', 'Yilan.csv', 'Yunlin.csv']
	cityname = ['彰化縣', '嘉義市', '嘉義縣', '新竹市', '新竹縣', '花蓮縣', '高雄市', '基隆市', '金門縣', '連江縣', '苗栗縣', '南投縣', '新北市', '澎湖縣', '屏東縣', '台中市', '台南市', '台北市', '台東縣', '桃園縣', '宜蘭縣', '雲林縣']

	income = pd.read_csv(filename[0], names = ['site_id', 'village', 'people', 'total', 'average', 'median', '25%', '75%', 'standard_deviation', 'variance']).drop(0).iloc[:-2, [0, 1, 4]]
	income.iloc[:, 0] = cityname[0] + income.iloc[:, 0].astype(str)
	for i in range(1, len(filename)):
		cityincome = pd.read_csv(filename[i], names = ['site_id', 'village', 'people', 'total', 'average', 'median', '25%', '75%', 'standard_deviation', 'variance']).drop(0).iloc[:-2, [0, 1, 4]]
		cityincome.iloc[:, 0] = cityname[i] + cityincome.iloc[:, 0].astype(str)
		income = pd.concat([income, cityincome], ignore_index = True)

	income = income.sort_values(by = ['site_id', 'village']).reset_index(drop = True)
	income.iloc[:, 2:] = income.iloc[:, 2:].apply(pd.to_numeric)

	return income





# Read Referendum 13 File
def readReferendum13():
	referendum13 = pd.read_csv('referendum13.csv', names = ['city', 'district', 'village', 'place', 'total_people', 'Agree', 'Disagree', 'Useful', 'Useless', 'total_tickets']).iloc[391:, [0, 1, 2, 5, 7]]
	referendum13['site_id'] = referendum13['city'].map(str) + referendum13['district']
	referendum13 = referendum13[['site_id', 'city', 'district', 'village', 'Agree', 'Useful']]

	referendum13 = referendum13.drop(referendum13.iloc[:, [1, 2]], axis = 1).sort_values(by = ['site_id', 'village']).reset_index(drop = True)
	referendum13.iloc[:, 2:] = referendum13.iloc[:, 2:].apply(pd.to_numeric)
	referendum13 = referendum13.groupby(['site_id', 'village']).sum()
	referendum13.to_csv('r13.csv')

	return referendum13





# Read Referendum 10 File
def readReferendum10():
	referendum10 = pd.read_csv('referendum10.csv', names = ['city', 'district', 'village', 'place', 'total_people', 'Agree', 'Disagree', 'Useful', 'Useless', 'total_tickets']).iloc[391:, [0, 1, 2, 5, 7]]
	referendum10['site_id'] = referendum10['city'].map(str) + referendum10['district']
	referendum10 = referendum10[['site_id', 'city', 'district', 'village', 'Agree', 'Useful']]

	referendum10 = referendum10.drop(referendum10.iloc[:, [1, 2]], axis = 1).sort_values(by = ['site_id', 'village']).reset_index(drop = True)
	referendum10.iloc[:, 2:] = referendum10.iloc[:, 2:].apply(pd.to_numeric)
	referendum10 = referendum10.groupby(['site_id', 'village']).sum()
	referendum10.to_csv('r10.csv')

	return referendum10





# Main Function
if __name__ == '__main__':
	education = readEducation()
	print(education.shape)
	merriage = readMerriage()
	print(merriage.shape)
	livingsituation = readLivingsituation()
	print(livingsituation.shape)
	family = readFamily()
	print(family.shape)
	population = readPopulation()
	print(population.shape)
	income = readIncome()
	print(income.shape)
	referendum13 = readReferendum13()
	print(referendum13.shape)
	print(referendum13.head(5))
	referendum10 = readReferendum10()
	print(referendum10.shape)
	print(referendum10.head(5))
	

	data = pd.merge(education, merriage, on = ['site_id', 'village'])
	print(data.shape)
	data = pd.merge(data, livingsituation, on = ['site_id', 'village'])
	print(data.shape)
	data = pd.merge(data, family, on = ['site_id', 'village'])
	print(data.shape)
	data = pd.merge(data, population, on = ['site_id', 'village'])
	print(data.shape)
	data = pd.merge(data, referendum13, on = ['site_id', 'village'])
	print(data.shape)
	data = pd.merge(data, referendum10, on = ['site_id', 'village'])
	print(data.shape)
	#data = pd.merge(data, income, on = ['site_id', 'village'])
	#print(data.shape)
	print(data.head(5))

	data.to_csv('data.csv')
