
DEFINE preprocess(path) returns data
{

        airline = load '$path' using PigStorage(',') 
            as (num:int, Year: int, Month: int, Carrier: chararray, Airport: chararray, Arr_flights: int, Arr_del15: int, Arr_delay:int);

	$data = foreach airline generate Arr_delay as delay, Year, Month, Carrier, Airport, Arr_flights, Arr_del15;
};

data_train = preprocess('/2018.csv');
rmf airline/train_1
store data_train into 'airline/2018_1' using PigStorage(',');
