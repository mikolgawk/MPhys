# MPhys

12.10
W mp_band_structures w opisie get_plot() dodałem ax.axis("off") i oczywix walnałęm range energii od -1eV do +1eV.
Ten plik z trenowaniem sieci powinien działać, ale samo trenowanie idzie narazie średnio, wydaje mi się, że trzeba zrobić random shuffle tych obrazków i potem je dać do trenowania.
Spróbuję przkonwertować resztę tego kodu w Matlabie na Pythona, możemy jednocześnie pobrać w pytę tych band structures z materialsproject i spróbować je wrzucić do tej sieci z Matlaba (jak już ogarniemy jak działa).
Wrzuciłem też trochę tych band structures z 2dmatpedia co Anupam używa w kodzie z modułu 2. Z tego co ogarniam to te jsony zamienia w png i potem w module 3 je segmentuje.

20.10
mp_band_structures.py pobiera band sturcutures dla materiałów z MP.
db-mp.py pobiera różne properties materiałów, jest to potem potrzebne do module 2.
Module_2_mine jako input file ma band structures i db-mp.py i zamienia JSONy w PNG plus jeszcze jeden output file wywala.
