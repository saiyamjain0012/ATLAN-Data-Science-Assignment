# ATLAN-Data-Science-Assignment
Working with the Geo coordinates taken from OSM to know the commercial centers of Delhi

 I have divided this solution in 4 major parts- 
 1. Making the query (Based on user selection)
 2. Getting the data through the web query 
 3. Cleaning and plotting the data i.e. POI retrieved 
 4. Making the clusters ( Used 2 methods- DBSCAN and KMEANS ) 
     
     
     
I have made this notebook in a dashboard style manner, due to the rendering incmpatibility and to ensure environment flexibility, I could not add a UI dashboard or a TABLEAU dashboard in this jupyter notebook. 
But I have added easy to use interactive widgets in order to make same like a dashboard.

For the definition of a commerical centre, I have assumed that there will be a collection of places like-
1. schools 
2. libraries
3. banks
4. hospital
5. cinema halls 
6. marketplace
7. pubs 
8. cafe
9. parking areas
10. fuel stations 
11. food courts 
12. restaurants 
13. bus stations 

These are the 'amenities' available to fetch from OSM (Open Street Map) [Source: https://wiki.openstreetmap.org/wiki/Key:amenity]
<br>
<b>A high occurance of a combination of these places around a particular locality would mean that that is a commercial centre of the city.</b> 
<br>
So my plan of action in this solution is- <br>
Select categories of places to look at -> Plot them on the map -> Make clusters where they occur close to each other -> Calculte and plot the centre of those localities, thus getting the commercial centre.

Note- Place the delhi map files in the same folder as the .ipynb files to render them into the notebook. Also the interactive widgets used in the solution maynot render properly in the .html file and so I have added the snapshots of those. They will work perfectly when openend in the .ipynb file though.
In case of any other issue or query do let me know at saiyamjain0012@gmail.com
