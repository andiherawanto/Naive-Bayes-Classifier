#LOAD PACKAGE#
library(tm)
library(NLP)
library(stringr)
library(caret)
library(dplyr)
library(naivebayes)
library(katadasaR)
library(tau)
library(parallel)
library(e1071)

setwd('D:/Kuliah/SKRIPSI/Andi/program/ruang kerja')

#Ambil data#
dok=read.csv('data andi.csv',sep=';',header = TRUE)
glimpse(dok)

#Merubah csv ke Vector Corpus#
corpusdok=Corpus(VectorSource(dok$dokumen))
inspect(corpusdok)

###LANGKAH PREPROCESSING###
#Cleaning hapus URL#
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
dok_URL <- tm_map(corpusdok, content_transformer(removeURL))
inspect(dok_URL)

remove.mention <- function(x) gsub("@\\S+", "", x)
dok_mention <- tm_map(dok_URL, remove.mention)
inspect(dok_mention)

remove.hashtag <- function(x) gsub("#\\S+", "", x)
dok_hashtag <- tm_map(dok_mention, remove.hashtag)
inspect(dok_hashtag)

remove.emoticon <- function (x) gsub("[^\x01-\x7F]", "", x)
dok_emoticon <- tm_map(dok_hashtag,remove.emoticon)
inspect(dok_emoticon)

remove.code <- function (x) gsub("<\\S+","",x)
dok_code <- tm_map(dok_emoticon,remove.code)
inspect(dok_code)

#remove punctuation#
dok_punctuation <- tm_map(dok_code, content_transformer(removePunctuation))
inspect(dok_punctuation)

#remove number#
dok_nonumber <- tm_map(dok_punctuation, content_transformer(removeNumbers))
inspect(dok_nonumber)

#case folding#
dok_casefolding <- tm_map(dok_nonumber, content_transformer(tolower))
inspect(dok_casefolding)

#remove duplicate character
remove.char <- function (x) gsub("([[:alpha:]])\\1{2,}", "\\1",x)
dok_char <- tm_map(dok_casefolding,remove.char)
inspect(dok_char)

#remove whitespace#
dok_whitespace<-tm_map(dok_char,stripWhitespace)
inspect(dok_whitespace)

#Slang Word
#load slangword#
slang <- read.csv("slangword andi.csv", header=T)
old_slang <- as.character(slang$old) 
new_slang <- as.character(slang$new)
slangword <- function(x) Reduce(function(x,r) gsub(slang$old[r],slang$new[r],x,fixed=F), seq_len(nrow(slang)),x)
dok_slangword <- tm_map(dok_whitespace,slangword)
inspect(dok_slangword)

#STEMMING#
stem_text<-function(text,mc.cores=1)
{
  stem_string<-function(str)
  {
    str<-tokenize(x=str)
    str<-sapply(str,katadasaR)
    str<-paste(str,collapse = "")
    return(str)
  }
  x<-mclapply(X=text,FUN=stem_string,mc.cores=mc.cores)
  return(unlist(x))
}
dok_stemming<-tm_map(dok_slangword,stem_text)
dok_stemming<-tm_map(dok_stemming,stripWhitespace)
inspect(dok_stemming)

#remove stopwords#
swindo<-as.character(readLines("stopword andi.csv"))
dok_stopword<-tm_map(dok_slangword,removeWords,swindo)
dok_stopword<-tm_map(dok_stopword,stripWhitespace)
inspect(dok_stopword)

#Matrix representation#
dtm<-DocumentTermMatrix(dok_stopword)
inspect(dtm)
m<-as.matrix(dtm)

#Save Data Naive Bayes#
df<-data.frame(dok$kategori,text=unlist(sapply(dok_stopword,`[`)), stringsAsFactors=F)
write.csv(df,file="hasil preprocessing data skripsi.csv")

#Partitioning#
df.train <- df[1:422,]
df.test <- df[423:528,]
dtm.train <- dtm[1:422,]
dtm.train
dtm.test <- dtm[423:528,]
dtm.test

corpus.clean.train <- dok_stopword[1:422]
corpus.clean.train
corpus.clean.test <- dok_stopword[423:528]
corpus.clean.test

#Featured Selection#
fivefreq <- findFreqTerms(dtm.train,1)
length((fivefreq))
dtm.train.nb_1 <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dtm.train.nb_1
dim(dtm.train.nb_1)
dtm.test.nb_1<- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))
dtm.test.nb_1
dim(dtm.test.nb_1)

#Boolan Naive Bayes#
convert_countNB <- function(x) 
{
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

#Naive Bayes Model#
trainNB <- apply(dtm.train.nb_1, 2, convert_countNB)
testNB <- apply(dtm.test.nb_1, 2, convert_countNB)

#Training#
classifier <- naiveBayes(trainNB, df.train$dok.kategori, laplace = 1)

#Use the NB classifier we built to make predictions on the train set and the test set#
predtraining <- predict(classifier, trainNB)
tabelprobtrain<- predict(classifier, trainNB,"raw")


prediksitrain<-as.character(predtraining)
pr<-data.frame(dok$kategori[1:422],prediksitrain,text=unlist(sapply(dok_stopword[1:422],`[`)), stringsAsFactors=F)
write.csv(pr,file="prediksi training.csv")




predtesting <- predict(classifier, testNB)
tabelprobtest<- predict(classifier, testNB, "raw")


prediksitest<-as.character(predtesting)
ppr<-data.frame(dok$kategori[423:528],prediksitest,text=unlist(sapply(dok_stopword[423:528],`[`)), stringsAsFactors=F)
write.csv(ppr,file="prediksi testing.csv")

#Create a truth table by tabulating the predicted class labels with the actual predicted class labels with the actual class labels#
NB_tabletraining=table("Prediction"=predtraining,"Actual"= df.train$dok.kategori)
NB_tabletraining

NB_tabletesting=table("Prediction"= predtesting, "Actual" = df.test$dok.kategori)
NB_tabletesting

#DTM#
dok_makanan<-Corpus(VectorSource(df[df$dok.kategori=="makanan",2]))
dok_transportasi<-Corpus(VectorSource(df[df$dok.kategori=="transportasi",2]))
tdm1<-TermDocumentMatrix(dok_makanan)
tdm2<-TermDocumentMatrix(dok_transportasi)
m1<-as.matrix(tdm1)
m2<-as.matrix(tdm2)
v1<-sort(rowSums(m1),decreasing = TRUE)
v2<-sort(rowSums(m2),decreasing = TRUE)
d1<-data.frame(word=names(v1),freq=v1)
d2<-data.frame(word=names(v2),freq=v2)
head(d1,20)
head(d2,20)

#Word Cloud#
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
set.seed(100)
w_makanan<-wordcloud2(d1,size = 1,fontFamily = 'Segoe UI',color = "random-dark")
w_transportasi<-wordcloud2(d2,size = 1,fontFamily = 'Segoe UI',color = "random-dark")
