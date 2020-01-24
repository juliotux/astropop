print('Digite inteiros, cada qual seguido de ENTER; ^D ou ^Z para finalizar')
total = 0
count = 0
"""
while True:
    try:
        line = input('inteiro: ')
        if line:
            numero = int()
            total += numero
            count += 1
    except ValueError as err:
        print (err)
        continue
    except EOFError:
        break
if count:
    print('count =',count, 'total =',total,' média =',total/count)
"""

while True:
    try:
        line = input()
        if line:
            try:
                numero = int(line)
            except ValueError as err:
                print (err)
                continue
            total += numero
            count += 1  
    except EOFError:
        break
if count:
    print('count =',count, 'total =',total,' média =',total/count)
  
  