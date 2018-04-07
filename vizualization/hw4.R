library(ggplot2)
library(ggmap)
library(maps)
library(mapdata)
library(tidyr)
library(dplyr)

data_okcupid <- read.csv("/Users/antoinemarthey/Desktop/profiles.csv",header = TRUE,sep = ",")

usa <- map_data("usa")
w2hr <- map_data("world2Hires")
states <- map_data("state")

new_data <- data_okcupid %>% separate(location, c('City','State'), sep =",")

new_data$State <- trimws(new_data$State)
calcul <- new_data %>% group_by(State) %>% summarize(Frequence = n())

colnames(calcul)[1] <-"region"
new_data2 <- inner_join(states, calcul, by="region")


usa <- map_data("usa") 
ggplot() + geom_polygon(data = usa, aes(x=long, y = lat, group = group)) + 
  coord_fixed(1.3)

#ggplot(data = states) + 
#  geom_polygon(aes(x = long, y = lat, group = group), color = "white") + 
#  coord_fixed(1.3) +
#  guides(fill=FALSE)  

ca_df <- subset(states, region == "california")
counties <- map_data("county")
ca_county <- subset(counties, region == "california")

ca_base <- ggplot(data = ca_df, mapping = aes(x = long, y = lat, group = group)) + 
  coord_fixed(1.3) + 
  geom_polygon(color = "black", fill = new_data2$Frequence)
ca_base

ca_base + theme_nothing() + 
  geom_polygon(data = ca_county, fill = new_data2$Frequence, color = "white") +
  geom_polygon(color = "black", fill = new_data2$Frequence)  

ditch_the_axes <- theme(
  axis.text = element_blank(),
  axis.line = element_blank(),
  axis.ticks = element_blank(),
  panel.border = element_blank(),
  panel.grid = element_blank(),
  axis.title = element_blank())

elbow_room1 <- ca_base + 
  geom_polygon(data = new_data2, aes(fill = new_data2$Frequence), color = "white") +
  geom_polygon(color = "black", fill = new_data2$Frequence) +
  theme_bw() +
  ditch_the_axes