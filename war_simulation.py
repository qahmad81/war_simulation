"""
numpy test
Created on Mon Oct 12 19:08:33 2020
@author: ahmad odeh
Test project learn numpy with big data to find how numpy manage to process with short time
Population simulation
"""
import numpy as np
import time as t

# soldier cols dont change it
cols_num = 11
# health of soldier
hps = (80,85,90,95,100,105,110,115,120)
# number of soldier in army 1 you can try (10, 10000, 50000 ..)
army1size = 1000
# number of soldier in army 2
army2size = 1000
# round of 2 attack from each army and rest turn
rounds = 4
#name of soldier cols
dt = np.dtype([('number','i2'), ('hp','i2'), ('attack','i2'), ('hit_damage','i2'), ('explode_damage','i2'), ('fire_damage','i2'), ('hit_def','i2'), ('explode_def','i2'), ('fire_def','i2'), ('energy_cost','i2'), ('energy','i2')])

def create_army(thesize):
    # create army array
    ary = np.arange(cols_num * thesize).reshape((thesize, cols_num))
    # set dtype
    ary.astype(dt)
    # soldier id used as primary key
    ary[:,0] = np.arange(thesize)
    # set soldier health as random of choice of hps
    ary[:,1] = np.random.choice(hps, size=(thesize))
    # set attak as random from 15 to 35
    ary[:,2] = np.random.randint(low=15, high=35, size=(thesize))
    # set hit, explode and fire attak attr and hit, explode and fire defince attr from 0 to 20
    ary[:,3:9] = np.random.randint(low=0, high=20, size=(thesize,6))
    # set attak energy cost random from 7 to 18
    ary[:,9] = np.random.randint(low=7, high=18, size=thesize)
    # set energy as fixed value of 100
    ary[:,10] = 100
    return ary

def print_sample(data, rows_num):
    data_rows = len(data)
    rows_keys = np.unique(np.random.randint(low=0, high=data_rows, size=rows_num))
    print(data[rows_keys])

# am sorry i cant find how merge tow array depend on column in numpy
# i will replace this when i find it
# so i post a question on stackoverflow to ask for help in this link
# https://stackoverflow.com/questions/64513436/in-python-with-numpy-how-can-i-update-array-from-another-array-depend-on-column
def update_ary_with_ary(source, updates):
    for x in updates:
        index_of_col = np.argwhere(source[:,0] == x[0])
        source[index_of_col] = x

def soldiers_rest(soldiers):
    soldiers_num = len(soldiers)
    # restore energy from 7 to 20 point
    soldiers[:,10] += np.random.randint(low=7, high=20, size=(soldiers_num))
    is_morethan100 = soldiers[:,10] > 100
    soldiers[is_morethan100,10] = 100

    # restore health from 4 to 14 point
    soldiers[:,1] += np.random.randint(low=3, high=8, size=(soldiers_num))
    is_morethan120 = soldiers[:,1] > 120
    soldiers[is_morethan120,1] = 120
    

def print_report(report, title):
    print('------- ', title, ' ----------')
    for x, y in report.items():
        print(x, ": ", y)


def attak_action(attacker, defencer): 
    ct = t.perf_counter()
    
    # to make report of attack
    attak_info = {}
    
    # in first filter layer death soldiers does not join in attak by sure
    attacker_remove_death = attacker[:,1] > 0
    attacker_layer = attacker[attacker_remove_death]
    
    # and sure death defencer will not attaks  
    defencer_remove_death = defencer[:,1] > 0
    defencer_layer = defencer[defencer_remove_death]
    
    # before attack they should be shuffle like real life 
    np.random.shuffle(attacker_layer)
    np.random.shuffle(defencer_layer)

    # from attaker the soldiers that have less than 25 of enery will not attack
    # they will recaver there energy and HP
    attacker_need_rest = attacker_layer[:,10] < 25
    attacker_need_heal = attacker_layer[:,1] < 20
    attacker_need_out = attacker_need_rest | attacker_need_heal
    attacker_rest = attacker_layer[attacker_need_out]
    
    attak_info['attaker_rest'] = len(attacker_rest)
    if attak_info['attaker_rest'] >= 1:
        soldiers_rest(attacker_rest)    
        # so the rest soldiers not join in attack
        attacker_layer = attacker_layer[np.invert(attacker_need_rest)]
        # store rest to source
        update_ary_with_ary(attacker, attacker_rest)
    
    # so how match remaning on each attacker and defencer after remove death fron tow side
    attacker_len = len(attacker_layer)
    defencer_len = len(defencer_layer)
    attack_len = attacker_len
    
    attak_info['attaker'] = attacker_len
    attak_info['defencer'] = defencer_len
    

    # if size of attacker and defencer is different the large one will resize as small one and the overflow will take energy
    if attacker_len != defencer_len:
        if attacker_len > defencer_len:
            # in this case attaker is more than defencer so they should resize to equal defencer size
            # but first the fighter they not join in attak should take profit 
            # so rest attaker take 10 point more at energy
            attacker_rest = attacker_layer[defencer_len:]
            soldiers_rest(attacker_rest) 
            # but if they have more than 100 as energy they should set to 100
            is_morethan100 = attacker_rest[:,10] > 100
            attacker_rest[is_morethan100,10] = 100
            # store nomber of rested attacker in the attack report 
            attak_info['attaker_rest'] += len(attacker_rest)
            # update rest of attaker to source (save)
            update_ary_with_ary(attacker, attacker_rest)
            # resise attaker to equal defencer size
            attacker_layer = attacker_layer[:defencer_len]
            attack_len = defencer_len
        else:
            # same of attaker in case defencer more than attaker
            defencer_rest = defencer_layer[attacker_len:]
            soldiers_rest(defencer_rest)
            attak_info['defencer_rest'] = len(defencer_rest)
            update_ary_with_ary(defencer, defencer_rest)
            defencer_layer = defencer_layer[:attacker_len]
            attack_len = attacker_len

    # create attak array calculation
    attack = np.zeros((attack_len, 10))
    # set attak id
    attack[:,0] = np.arange(attack_len)
    # set sum of hit, explode and fire attr to use it next as attack_atr_sum
    attack[:,1] = np.sum(attacker_layer[:,3:6], axis=1) + 1
    #calculate hit, explode and fire damage as:  ( hit / attack_atr_sum ) * damage * ( energy / 100 )
    attack[:,2] = (attacker_layer[:,3] / attack[:,1]) * attacker_layer[:,2] * (attacker_layer[:,10] / 100)
    attack[:,3] = (attacker_layer[:,4] / attack[:,1]) * attacker_layer[:,2] * (attacker_layer[:,10] / 100)
    attack[:,4] = (attacker_layer[:,5] / attack[:,1]) * attacker_layer[:,2] * (attacker_layer[:,10] / 100)
    # calculate attack amount on each hit, explode and fire with defencer def atr as this formula
    # final_hit_damage = attaker_hit_damage * (20 / (25 + (defencer_hit_attr - attaker_hit_attr)))
    attack[:,5] = attack[:,2] * 20 / (25 + defencer_layer[:,6] - attacker_layer[:,3])
    attack[:,6] = attack[:,3] * 20 / (25 + defencer_layer[:,7] - attacker_layer[:,4])
    attack[:,7] = attack[:,4] * 20 / (25 + defencer_layer[:,8] - attacker_layer[:,5])
    # final damage that will defencer take as formula sum of fanal attr damage
    attack[:,8] = np.sum(attack[:,5:8], axis=1)
    
    attack = np.round(attack).astype(int)
    damage = np.round(attack[:,8]).astype(int)
    # decrease energy with energy cost for attaker
    attacker_layer[:,10] -= attacker_layer[:,9]

    # if Energy less than 0 set it at 0    
    is_lowthanzero = attacker_layer[:,10] < 0
    attacker_layer[is_lowthanzero,10] =0

    # defencer will take damage and energy suffer as ratio or health loss
    # (damage / health) * current_energy
    defencer_layer[:,10] = defencer_layer[:,10] - (damage / defencer_layer[:,1]) * defencer_layer[:,10]

    # defencer take calculated damage
    defencer_layer[:,1] -= damage
    #print(damage)
    
    attak_info['total_damage'] = np.int64(sum(damage))

    # if HP less than 0 set it at 0    
    is_lowthanzero = defencer_layer[:,1] < 0
    defencer_layer[is_lowthanzero,1] = 0
    # if Energy less than 0 set it at 0    
    is_lowthanzero = defencer_layer[:,10] < 0
    defencer_layer[is_lowthanzero,10] = 0
    
    # store new death from defencer in attack report
    is_zero = defencer_layer[:,1] == 0
    defencer_death = defencer_layer[is_zero,0]
    attak_info['new_death'] = len(defencer_death)

    # transfer attacker and defencer data from this attack to source army array
    update_ary_with_ary(attacker, attacker_layer)
    update_ary_with_ary(defencer, defencer_layer)
    pt = t.perf_counter()
    attak_info['attack_calculation_time'] = pt-ct
    return attak_info

def armstatus(army):
    health = army[:,1].copy()
    status = {}
    status['Total'] = len(health)
    Remans = health[health > 0]
    status['Remans'] = len(Remans)
    status['Health'] = round(sum(Remans) / status['Remans'])
    status['Poor'] = len(Remans[Remans < 50])
    status['Death'] = len(health[health == 0])
    status['Good'] = len(health[health >= 50])
    return status


column_names = ['hp', 'attack', 'break hit', 'explode hit', 'fire hit', 'break resistance', 'explode resistance', 'fire resistance']

# store general report
general_report = {}

ct = t.perf_counter()
all_start_time = ct 
army1 = create_army(army1size)
pt = t.perf_counter()
general_report['army1_create_time'] = pt-ct

ct = t.perf_counter()
army2 = create_army(army2size)
pt = t.perf_counter()
general_report['army2_create_time'] = pt-ct

all_rounds = 5*rounds
for i in range(all_rounds):
    if i%10 == 0 or i%10 == 2 or i%10 == 6 or i%10 == 8:
        print_report(attak_action(army1, army2), 'Attack report for Army1 attak Army2')
    elif i%10 == 1 or i%10 == 3 or i%10 == 5 or i%10 == 7:
        print_report(attak_action(army2, army1), 'Attack report for Army2 attak Army1')
    else:
        # rest for army 1
        remove_death = army1[:,1] > 0
        army1_layer = army1[remove_death]
        soldiers_rest(army1_layer)
        update_ary_with_ary(army1, army1_layer)
        print_report(armstatus(army1), "Status of Army1 after rest")
        print('Sample of army 1 data')
        print_sample(army1,4)

        # rest for army 2
        remove_death = army2[:,1] > 0
        army2_layer = army2[remove_death]
        soldiers_rest(army2_layer)
        update_ary_with_ary(army2, army2_layer)
        print_report(armstatus(army2), "Status of Army2 after rest")
        print('Sample of army 2 data')
        print_sample(army2,4)
    
army1_status = armstatus(army1)
army2_status = armstatus(army2)

if army1_status['Remans'] > army2_status['Remans']:
    general_report['winer'] = 'Army 1'
elif army1_status['Remans'] < army2_status['Remans']:
    general_report['winer'] = 'Army 2'
else:
    if army1_status['Health'] > army2_status['Health']:
        general_report['winer'] = 'Army 1'
    elif army1_status['Health'] < army2_status['Health']:
        general_report['winer'] = 'Army 2'
    else:
        general_report['winer'] = 'No win'

all_end_time = t.perf_counter()
general_report['all_time'] = all_end_time - all_start_time
print_report(general_report, "Final result")
