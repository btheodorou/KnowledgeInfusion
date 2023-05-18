from sympy.logic.boolalg import to_dnf
from sympy import symbols

#inpatient
l10, l11, l12, l13, l14, l15 = symbols('1610 1611 1612 1613 1614 1615')
z14, z93, o96, o97, o58, o66, z88 = symbols('514 93 1196 1297 1158 1366 288')
#the line below outputs constraints for the inpatient model in dnf form
print(to_dnf((z14 >> z93) & (o96 >> z93) & (o97 >> o58) & (o66 >> z88) & (l14 >> ~l15) & ((l10 & ~l11 & ~l12 & ~l13) | (~l10 & l11 & ~l12 & ~l13) | (~l10 & ~l11 & l12 & ~l13) | (~l10 & ~l11 & ~l12 & l13)), simplify=True, force=True))


#l17, l18, l19, l20, l21, l22, l23 = symbols('1817 1818 1819 1820 1821 1822 1823')
#o69, o97, z12, z90 = symbols('1169 1596 512 1823')
##the line below outputs constraints for the outpatient model in dnf form
#print(to_dnf((o69 >> o97) & (o69 >> z12) & (o69 >> z90) & (l22 >> ~l23) & ((l17 & ~l18 & ~l19 & ~l20 & ~l21) | (~l17 & l18 & ~l19 & ~l20 & ~l21) | (~l17 & ~l18 & l19 & ~l20 & ~l21) | (~l17 & ~l18 & ~l19 & l20 & ~l21) | (~l17 & ~l18 & ~l19 & ~l20 & l21)), simplify=True, force=True))