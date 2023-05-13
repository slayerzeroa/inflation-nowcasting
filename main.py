import Core_Inflation as ci

core_inplation = ci.calculate()
core_inplation = core_inplation.reshape(2)

Kor_CPI, Kor_production = core_inplation[0], core_inplation[1]

print(f"한국의 예상 CPI 상승률은 {Kor_CPI}%, 예상 생산자 물가 상승률은 {Kor_production}% 입니다.")