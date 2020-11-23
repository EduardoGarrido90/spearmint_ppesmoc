rom scipy.stats import beta
from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.signal import argrelextrema

#Must be sampled from a GP.
def sample_objective_function(x):
        sin_length = uniform(10, 50)
        shift = uniform(0.0, 2.0*np.pi)
	funcs = np.array([np.sin,np.cos])
        target = funcs[np.random.randint(0,2)](x*sin_length.rvs() + shift.rvs())
        t_max = np.max(target)
        t_min = np.min(target)
        return (target-t_min)/(t_max-t_min)

def fixed_objective_function(x):
	target = np.sin(x*50)
	t_max = np.max(target)
	t_min = np.min(target)
	return (target-t_min)/(t_max-t_min)

def sinc_pattern(x):
	x[x==0] = np.nextafter(0,1)
	return np.sin(x*np.pi)/(np.pi*x)

#Algo se esta haciendo mal al dewarpear, no tiene sentido lo que aparece. Ya esta, al dewarpear no hace falta meter otra vez la transformacion de la funcion tonto!
ax = plt.subplot(111)
a, b = uniform(), uniform()
input_space = np.linspace(0, 1, 1000)

#objective = sample_objective_function(input_space)
objective = fixed_objective_function(input_space)

#The stationary function must have the same number of extremas in order to be valid.
#The other solutions are not feasible.
extrema_max_obj = argrelextrema(objective, np.greater)[0].shape[0]
extrema_min_obj = argrelextrema(objective, np.less)[0].shape[0]

ax.plot(input_space, objective, label='Objective')

#Probar a relajar verificando solo el maximo o relajando a un intervalo de +-1 el numero de extremos.
#Y debuggear esto.
#Asi hasta que salga el ejemplo de dewarpear un warping con beta pero imposible dos.
#Y si posible con generalizacion de la beta. Dos-Cuatro ejemplos de esto.
#Mas tarde hacer lo mismo con una funcion juguete y no sintetica. Ejemplos que no van debajo del codigo.
#Y como mucho un ejemplo real.
#Y ya estaria listo para Spearmint.
#Beta params
discovered = False
counter = 0
while not discovered:
	a_param = a.rvs()
	b_param = b.rvs()
	beta_fun = beta.cdf(input_space, a_param, b_param)
	warped_objective = fixed_objective_function(beta.cdf(input_space, a_param, b_param))
	e_w_max = argrelextrema(warped_objective, np.greater)[0].shape[0]
	e_w_min = argrelextrema(warped_objective, np.less)[0].shape[0]
	counter+=1
	print counter
	if e_w_max == extrema_max_obj and e_w_min == extrema_min_obj:
		discovered = True

#Now we have a warped objective with the same number of extremas than the original one.
ax.plot(input_space, beta.cdf(input_space, a_param, b_param), label='Beta')
ax.plot(input_space, warped_objective, label='Objective Beta Warped')
plt.legend(loc='best')
plt.show()

#Stationarity measures. 100 divisions of input spaces.
'''
div_i_s = input_space.reshape((1000,100))
div_obj = objective.reshape((1000,100))
div_beta_fun = beta_fun.reshape((1000,100))
div_warped_objective = warped_objective.reshape((1000,100))

#Measure no. 1. Statistics.
means_do = np.array([np.mean(div) for div in div_obj])
vars_do = np.array([np.var(div) for div in div_obj])
medians_do = np.array([np.median(div) for div in div_obj])

means_wo = np.array([np.mean(div) for div in div_warped_objective])
vars_wo = np.array([np.var(div) for div in div_warped_objective])
medians_wo = np.array([np.median(div) for div in div_warped_objective])
'''

#Measure no. 2. Augmented Dickey-Fuller test.
#adf_o = adfuller(objective, regression='ctt', maxlag=100)
#adf_wo = adfuller(warped_objective, regression='ctt', maxlag=100)

adf_ko = kpss(objective, regression='ct') #El kpss para esto es bastante mas fiable que el adfuller. Se podria jugar con el para estacionar las funciones no estacionarias con grids de parametros de betas. Lo ideal es conseguir una medida de estacionaridad mas avanzada pero whatever.

#Alguna vez se traga algo non stationary, pero esta bastante bien. Se podria combinar con regression='c' y sumar ambos coeficientes. No va mal.
adf_kwo = kpss(warped_objective, regression='ct')

#sub = np.sum(np.abs(warped_objective-objective))+np.var(np.abs(warped_objective-objective))

#Print results.
'''
print 'Original mean.'
print means_do
print 'Original var.'
print vars_do
print 'Original median'
print medians_do
print 'Warped mean'
print means_wo
print 'Warped var.'
print vars_wo
print 'Warped median'
print medians_wo

print 'Differences'
print np.abs(means_do-means_wo)
print np.abs(vars_do-vars_wo)
print np.abs(medians_do-medians_wo)


discrete_is = np.linspace(0, 1, 100)
ax = plt.subplot(111)
ax.plot(discrete_is, means_do, label='Means stationarity function')
ax.plot(discrete_is, means_wo, label='Means non-stationarity function')
plt.legend(loc='best')
plt.show()

'''
#Final results.
'''
print 'Adfuller obj value: ' + str(adf_o[0]) + '. p-value: ' + str(adf_o[1])
if adf_o[1] < 0.05:
	print 'Stationary with p-value threshold 0.05'
else:
	print 'Non-stationary with p-value threshold 0.05'

print 'Adfuller warped obj: ' + str(adf_wo[0]) + '. p-value: ' + str(adf_wo[1])
if adf_wo[1] < 0.05:
	print 'Stationary with p-value threshold 0.05'
else:
        print 'Non-stationary with p-value threshold 0.05'
'''
print 'KPSS obj value: ' + str(adf_ko[0]) + '. p-value: ' + str(adf_ko[1])
if adf_ko[0] < 0.03:
        print 'Stationary'
else:
        print 'Non-stationary'

print 'KPSS warped obj: ' + str(adf_kwo[0]) + '. p-value: ' + str(adf_kwo[1])
if adf_kwo[0] < 0.03:
        print 'Stationary'
else:
        print 'Non-stationary'

import pdb; pdb.set_trace();
#El grid search de aqui es muy cutre pero funciona como una demo, el objetivo es minimizar el test de estacionaridad.
#Este tendria que salir. Debuggear esto.
print 'Starting grid search to look for the best beta params to squash the input params in order to get a stationary function'
print 'The obj function is to maximize KPSS s.t. having the same number of local minimas and maximas'
a_grid = np.linspace(0.1,10,100)
b_grid = np.linspace(0.1,10,100)
adf_min_val = 20
k_min_val = 20
a_min_value = -1
b_min_value = -1
min_val = 1000000
min_distance = 100000
counter = 0
print 'Objective number maxs: ' + str(extrema_max_obj) + ', number mins: ' + str(extrema_min_obj)
print 'Warped number maxs: ' + str(argrelextrema(warped_objective, np.greater)[0].shape[0]) + ', number mins: ' + str(argrelextrema(warped_objective, np.less)[0].shape[0])
for a_value in a_grid:
	for b_value in b_grid:
		#wo = fixed_objective_function(beta.cdf(warped_objective, a_value, b_value))
		wo = beta.cdf(warped_objective, a_value, b_value)
		#adf_wog = adfuller(wo, regression='ctt', maxlag=100)
		#k_wog = kpss(wo, regression='ct')
		extrema_max_wo = argrelextrema(wo, np.greater)[0].shape[0]
		extrema_min_wo = argrelextrema(wo, np.less)[0].shape[0]
		distance = np.sum(np.abs(objective-wo))
		print 'Dewarped number maxs: ' + str(extrema_max_wo) + ', number mins: ' + str(extrema_min_wo)
		#min_val_i = np.sum(np.abs(wo-objective))
		#min_val_i = np.sum(np.abs(wo-objective))+np.var(np.abs(wo-objective))
		'''
		if min_val_i < min_val:
			min_val = min_val_i
			a_min_value = a_value
                        b_min_value = b_value
		
		if adf_wog[0] < adf_min_val:
			adf_min_val = adf_wog[0]
			a_min_value = a_value
			b_min_value = b_value
		'''
		'''
		if k_wog[0] < k_min_val and extrema_max_wo == extrema_max_obj and extrema_min_wo == extrema_min_obj:
                        k_min_val = k_wog[0]
                        a_min_value = a_value
                        b_min_value = b_value
		'''
		if distance < min_distance and extrema_max_wo == extrema_max_obj and extrema_min_wo == extrema_min_obj:
                        min_distance = distance
                        a_min_value = a_value
                        b_min_value = b_value
		counter+=1
		print counter

#print 'Results: Original adfuller min value: ' + str(adf_o[0])
print 'Results: Original KPSS min value: ' + str(adf_ko[0])
print 'Results best KPSS beta: ' + str(k_min_val)
#print 'Results of best beta, adfuller: ' + str(adf_min_val)
print 'Plotting...'

ax = plt.subplot(111)
a, b = uniform(), uniform()

ax.plot(input_space, objective, label='Objective')

#Beta params
#Arreglar esto.
dewarped = beta.cdf(warped_objective, a_min_value, b_min_value)
ax.plot(input_space, warped_objective, label='Warped Objective')
ax.plot(input_space, beta.cdf(input_space, a_min_value, b_min_value), label='Beta')
ax.plot(input_space, dewarped, label='Objective Dewarped')
plt.legend(loc='best')
plt.show()

#Hecho lo del test de estacionaridad, se va a comparar con la funcion estacionaria perfecta, la mas cercana en distancia, ( suma de restas ) gana.
#Para el ejemplo vale, pero seguro que hay una mejor forma de calcular la distancia entre dos funciones.
#Como observacion, la beta es capaz de hacer warp y convertir en estacionaria una funcion no estacionaria a ambos lados simetrica, vas a tener que pelear mas.
#2. Una vez tengas esto, encontrar una funcion para la cual la beta no da un buen valor. Estacionaria ambos lados mal.

#Revisar estos valores y ver cual puede ser la causa de que el resultado que de sea una funcion escalon, esa beta no mola. ( La estacionaridad es mala o rara ).
#Puede ser de la amplitud, dado que las funciones no estan entre 0 y 1 como la primera, ojo.
#Ejemplo sinc pattern.
input_space = np.linspace(-10,10,10000)
sinc_pat = np.sin(input_space*np.pi)/(np.pi*input_space) #No vale, el sinc pattern es bonito, pero es estacionario. Hay que encontrar otro ejemplo.
extrema_max_sinc = argrelextrema(sinc_pat, np.greater)[0].shape[0]
extrema_min_sinc = argrelextrema(sinc_pat, np.less)[0].shape[0]
ax = plt.subplot(111)
ax.plot(input_space, sinc_pat, label='Sinc')
plt.legend(loc='best')
plt.show()

#KPSS value
k_sinc = kpss(sinc_pat, regression='ct')

print 'KPSS value of sinc pattern: ' + str(k_sinc[0])
if k_sinc[0] < 0.03:
	print 'Sinc Pattern Non stationary'

print 'Starting Grid Search for stationarizing Sinc Pattern with Betas'

#No se mejora.
k_sinc_min_val = 2000
a_grid = np.linspace(0.1,10,100)
b_grid = np.linspace(0.1,10,100)
a_min_value = -1
b_min_value = -1
counter = 0
for a_value in a_grid:
        for b_value in b_grid:
		res = sinc_pattern(beta.cdf(input_space, a_value, b_value))
                k_i_sinc = kpss(res, regression='ct')
		extrema_max_res = argrelextrema(res, np.greater)[0].shape[0]
		extrema_min_res = argrelextrema(res, np.less)[0].shape[0]
		if k_i_sinc[0] < k_sinc_min_val and extrema_max_res == extrema_max_sinc and extrema_min_res == extrema_min_sinc:
                        k_sinc_min_val = k_i_sinc[0]
                        a_min_value = a_value
                        b_min_value = b_value
                counter+=1
                print counter

print 'Results: Original KPSS min value: ' + str(k_sinc[0])
print 'Results best KPSS beta: ' + str(k_sinc_min_val)
print 'Plotting...'

ax = plt.subplot(111)
ax.plot(input_space, sinc_pat, label='Sinc_Pattern')
beta_fun = beta.cdf(input_space, a_min_value, b_min_value)
ax.plot(input_space, beta_fun, label='Beta')
ax.plot(input_space, sinc_pattern(beta.cdf(input_space, a_min_value, b_min_value)), label='Sinc Pattern Beta Warped')
plt.legend(loc='best')
plt.show()

#Ejemplo sin x 2. Esto si es valido, pero es muy sencillo. No estacionaridad simetrica a ambos lados. Aqui hay que probar la beta. Lo mas probable es que funcione.
#Pues tampoco se mejora.
input_space = np.linspace(-5,5,10000)
x_cuad = np.sin(np.power(input_space,2)) #No vale, el sinc pattern es bonito, pero es estacionario. Hay que encontrar otro ejemplo.
extrema_max_x_cuad = argrelextrema(x_cuad, np.greater)[0].shape[0]
extrema_min_x_cuad = argrelextrema(x_cuad, np.less)[0].shape[0]
ax = plt.subplot(111)
ax.plot(input_space, x_cuad, label='sin x power')
plt.legend(loc='best')
plt.show()

k_xcuad = kpss(x_cuad, regression='ct')

print 'KPSS value of x_cuad: ' + str(k_xcuad[0])
if k_xcuad[0] < 0.03:
	print 'X_cuad Non stationary'

print 'Starting Grid Search for stationarizing X_Cuad with Betas'

k_sinc_min_val = 100
a_grid = np.linspace(0.1,10,100)
b_grid = np.linspace(0.1,10,100)
a_min_value = -1
b_min_value = -1
counter = 0
for a_value in a_grid:
        for b_value in b_grid:
                res = np.sin(np.power(beta.cdf(input_space, a_value, b_value),2))
		extrema_max_res = argrelextrema(res, np.greater)[0].shape[0]
		extrema_min_res = argrelextrema(res, np.less)[0].shape[0]
                k_i_sinc = kpss(res, regression='ct')
                if k_i_sinc[0] < k_sinc_min_val and extrema_max_x_cuad == extrema_max_res and extrema_min_x_cuad == extrema_min_res:
                        k_sinc_min_val = k_i_sinc[0]
                        a_min_value = a_value
                        b_min_value = b_value
                counter+=1
                print counter

print 'Results: Original KPSS min value: ' + str(k_xcuad[0])
print 'Results best KPSS beta: ' + str(k_sinc_min_val)
print 'Plotting...'

ax = plt.subplot(111)
ax.plot(input_space, x_cuad, label='X_cuad')
beta_fun = beta.cdf(input_space, a_min_value, b_min_value)
ax.plot(input_space, beta_fun, label='Beta')
ax.plot(input_space, np.sin(np.power(beta.cdf(input_space, a_min_value, b_min_value),2)), label='X_cuad Beta Warped')
plt.legend(loc='best')
plt.show()

#Estaria bien warpear una estacionaria dos veces y asi demostrar que la beta no vale, haria
#estos experimentos mas faciles. Esta el tema del periodo, no se muy bien que hacer con esto, porque claro, para estacionario es mejor una recta? Y luego con eso
#como se vuelve al original en Spearmint? Revisar el paper de la transformacion para ver que se hace. Es una cosa rara, leer de nuevo el paper.
#Una vez leido proceder con la warped original ( yo haria un doble warped para que se vea que no se puede volver ) y lo desharia con una generalizada beta.
#Luego a Spearmint.
#Y lo ultimo seria conseguir que estos ejemplos vayan bien. 
#Si hace falta preservar ciclos o periodos habria que combinar el KPSS con el num periodos e incluso se podria meter algo del Adfuller, sera por historias.
#3. Una vez hecho esto, encontrar una distribucion/funcion que si convierta bien los valores de entrada para hacer la funcion estacionaria.
#La beta generalizada es el primer enfoque mas obvio. Usar un subespacio de valores de parametros de la beta generalizada frente a la beta puede molar.
#Ver otras distribuciones, casos particulares y generales y ver cuando usar unos u otros.
#A mas parametros mas dificil el aprendizaje, tener en cuenta, otro enfoque es ir cambiando segun se aprende, eso para trabajo futuro, pero mola!
#4. Lo siguiente seria programarlo en Spearmint y obtener distintos valores para Optimizacion Bayesiana.
#5. Lo siguiente seria generalizar ambos enfoques de forma barata para que se puedan usar los dos enfoques. Aqui hay tema serio de metodos de simulacion.
