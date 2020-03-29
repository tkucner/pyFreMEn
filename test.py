import FreMEn as fr
import FreMEnVis as fr_vis
import pandas as pd
times = [0, 3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200, 46800, 50400, 54000,
         57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800, 86400, 90000, 93600, 97200, 100800, 104400, 108000,
         111600, 115200, 118800, 122400, 126000, 129600, 133200, 136800, 140400, 144000, 147600, 151200, 154800, 158400,
         162000, 165600, 169200, 172800, 176400, 180000, 183600, 187200, 190800, 194400, 198000, 201600, 205200, 208800,
         212400, 216000, 219600, 223200, 226800, 230400, 234000, 237600, 241200, 244800, 248400, 252000, 255600, 259200,
         262800, 266400, 270000, 273600, 277200, 280800, 284400, 288000, 291600, 295200, 298800, 302400, 306000, 309600,
         313200, 316800, 320400, 324000, 327600, 331200, 334800, 338400, 342000, 345600, 349200, 352800, 356400, 360000,
         363600, 367200, 370800, 374400, 378000, 381600, 385200, 388800, 392400, 396000, 399600, 403200, 406800, 410400,
         414000, 417600, 421200, 424800, 428400, 432000, 435600, 439200, 442800, 446400, 450000, 453600, 457200, 460800,
         464400, 468000, 471600, 475200, 478800, 482400, 486000, 489600, 493200, 496800, 500400, 504000, 507600, 511200,
         514800, 518400, 522000, 525600, 529200, 532800, 536400, 540000, 543600, 547200, 550800, 554400, 558000, 561600,
         565200, 568800, 572400, 576000, 579600, 583200, 586800, 590400, 594000, 597600, 601200]
states = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
          1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1,
          1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

data={'times':times,'states':states}
df = pd.DataFrame(data)
FR=fr.FreMen()
FR.add_observations(time_series=df)
FR.update()
p,reconstruction_time=FR.reconstruct()
fr_vis.show(times=times,states=states,probabilities=p,reconstruction_times=reconstruction_time)