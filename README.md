Βάζουμε όλα τα excel σε ένα και αφαιρούμε τις πανομοιώτυπες εγκραφές
Κρατάμε Ημερομηνία εκδοσης άδειας -> μήνες έγκρισης & διάρκια (σε μήνες)
Κρατάμε μόνο περιφερειακή ενότητα
Κρατάμε μέγιστη ισχύ
Κρατάμε τεχνολογία -> one hot encoding

Minor modifications to the source excel files because some fields' names changed
For the duplicates we kept only the last record (we treat it as a modification not as a new permit)

Analysis on duplicates

EDA ideas:
* Plot MW timeseries trend (line)
* Plot MW per region (bar)
* Plot MW per technology (bar)
* 
* Plot number of permits per month & week trend (line)
* Plot number of permits per region (bar)
* Plot number of permits per technology (bar)
* 
* Plot average permit duration in months per region (bar)
* Plot average permit acceptance in months per region (bar)
* Plot technology in time (stacked-bar)
* Plot regions in time (stacked-bar)
* 
* C02 price