# Описание работы скриптов

## Валидация

### prepare_data
Приводит судовые измерения к единому и удобному для анализа формату.

### plot_validation

### calculate_\<Датасет\>_validation

Для каждого снимка из **Датасета** ищет подходящие по времени измерения с возможной ошибкой +- 15 минут. Из найденных измерений выбирает первое такое, в котором есть данные , так что с почти по-секундным сравниваем каждый снимок обрабатывается значительно дольше

# Описание ресурсов и результатов

## Валидация

### Данные_валидации
 ext
 число на конце
#### raw

Данные валидации без добавления дистанции до берега

### Наземеные_измерения

* raw - полученные данные датчиков в том виде, в котором они были предоставлены для анализа
* ready -  приведённые к единому и удобному для анализа формату данные датчиков
  * *_cut\<N\> - данные, в которых убраны некоторые записи таким образом, чтобы промежутки между измерениями были не менее чем **N** минут. Это сделано для ускорения вычислений: . А точности это не приносит, так как всё равно берётся первое найденное