Βάζουμε όλα τα excel σε ένα και αφαιρούμε τις πανομοιώτυπες εγκραφές
Κρατάμε Ημερομηνία εκδοσης άδειας -> μήνες έγκρισης & διάρκια (σε μήνες)
Κρατάμε μόνο περιφερειακή ενότητα
Κρατάμε μέγιστη ισχύ
Κρατάμε τεχνολογία -> one hot encoding

Minor modifications to the source excel files because some fields' names changed
For the duplicates we kept only the last record (we treat it as a modification not as a new permit)

Analysis on duplicates

EDA ideas:
* Plot MW timeseries trend (line) - DONE
* Plot MW per region (bar) - DONE
* Plot MW per technology (bar) - DONE
* 
* Plot number of permits per month & week trend (line) - DONE
* Plot number of permits per region (bar) - DONE
* Plot number of permits per technology (bar) - DONE
* 
* Plot average permit duration in months per region (bar) - DONE
* Plot average permit acceptance in months per region (bar) - DONE
* Plot average permit duration in months per technology (bar) - DONE
* Plot average permit acceptance in months per technology (bar) - DONE
* 
* Plot technology in time (line) - DONE
* Plot regions in time (line) - DONE
* 
* C02 price - DONE
* Wind speeds