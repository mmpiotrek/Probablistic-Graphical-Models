# Projekt
Projekt zaliczeniowy "Przechodnie" w ramach przedmiotu Sztuczna Inteligencja w Robotyce opiera się na grafie czynników. Projekt polega na śledzeniu, a dokładniej porównywaniu osób z następujących po sobie zdjęć. Każda osoba, która po raz pierwszy pojawiła się na monitoringu reprezentowana jest przez wartość -1. Osoby które już wcześniej zostały uchwycone przyjmują unikatowe stałe wartości równe lub większe od 0.

Na początku wczytywane są zdjęcia z monitoringu, z których następnie wycinane są osoby (przechodnie). Informacje o tym gdzie znajdują się osoby są wczytywane z pliku bboxes.txt. Następnie ze zdjęć przechodniów generowane są:
- histogram w skali szarości,
- histogram odcienia światła (hue),
- histogram nasycenia kolorów (saturation),
- stosunek szerokości do wysokości osoby.
- 
Uzyskane dane porównywane są z danymi osób z poprzedniego zdjęcia. W ten sposób powstaje macierz podobieństwa (hist_avarage o wymiarach 1x(ilość osób na zdjęciu + 1)). Macierz ta określa na ile osoby z dwóch kolejnych zdjęć są do siebie podobne. Jeżeli podobieństwo wynosi poniżej 65% osoba jest uznawana za nową. Każdy Bounding Box (czyli fragment zdjęcia z przechodniem) tworzy nowy węzeł (ang. node) w grafie czynników. Macierz podobieństwa dodawana jest jako krawędź w grafie i łączy się z tylko z odpowiadającym jej węzłem.

Jeżeli na jednym zdjęciu znajduje się więcej niż jedna osoba, wówczas powstają połączenia pomiędzy wszystkimi węzłami, a czynnikiem jest kwadratowa macierz zajętości (fp_matrix o boku wynoszącym: ilość osób na poprzednim zdjęciu + 1), dzięki której do każdej osoby z poprzedniego zdjęcia może być przypisana maksymalnie jedna osoba ze zdjęcia aktualnego. Finalnie jako wynik otrzymujemy słownik, z którego odczytujemy wartości za pomocą klucza, a następnie odejmujemy od nich 1, ponieważ funkcja BeliefPropagation zwraca wartości większe lub równe od 0.
