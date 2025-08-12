##extracting ASV table:
#Quality check: R, xml2, httpuv, latticeExtra, dada2
# China Data
#make sure to change the path here to the path of your data
fastq_path1 <- "C:/Users/YLS/Documents/IBD1356/data_IBD1356"

#Assign Sample Names:
library(dada2)
library(phyloseq)
library(ggplot2)
# Create RDS output folder
output_dir <- "C:/Users/YLS/Documents/IBD1356/rds_outputs_IBD1356"
if (!dir.exists(output_dir)) dir.create(output_dir)
fnFs1 <- sort(list.files(fastq_path1, pattern=".fastq", full.names = TRUE))
#fnRs1 <- sort(list.files(fastq_path1, pattern="_2.fastq", full.names = TRUE))
fnFs1
#fnRs1

# Extract sample names, assuming filenames have format: SAMPLENAME_XXX.fastq

sample.names <- sapply(strsplit(basename(fnFs1), "\\.") , `[`, 1)
#in case of paired end we use:
#sample.names <- sapply(strsplit(basename(fnFs1), "_"), `[`, 1)
sample.names

#Plot Quality Control charts:
#plotQualityProfile(fnFs1[1:2])
#plotQualityProfile(fnRs1[1:4])

#Assign names of the Filtered sequences:
#in case of single end:
filtFs1 <- file.path(fastq_path1, "filter", paste0(sample.names, "_filt.fastq.gz"))
#in case of paired end we use:
#filtFs1 <- file.path(fastq_path1, "filter5", paste0(sample.names, "_1_filt.fastq.gz"))
#filtRs1 <- file.path(fastq_path1, "filter5", paste0(sample.names, "_2_filt.fastq.gz"))
#filtFs1
#filtRs1
names(filtFs1) <- sample.names
sample.names
#in case of paired end we run also:
#names(filtRs1) <- sample.names

#filtRs1
filtFs1
#Trimming of the sequences:
#out <- filterAndTrim(fnFs1, filtFs1,fnRs1,filtRs1,truncLen =(240,220)  ,maxN=0, maxEE=c(2,2), truncQ=2, rm.phix=TRUE, compress=TRUE, multithread=TRUE, minQ=20, verbose=TRUE)

# This will spit out the number of the input and output reads.
out1 <- filterAndTrim(
  fnFs1, filtFs1,
  truncLen = 150,      # Match the SHORTEST read length across samples
  maxN = 0,            # Keep strict for ambiguous bases
  maxEE = Inf,         # Disable EE filtering temporarily (for debugging)
  truncQ = 0,          # Disable quality truncation (for debugging)
  minLen = 50,         
  rm.phix = FALSE,     # Disable PhiX filtering (in case it’s misapplied)
  compress = TRUE,
  multithread = TRUE
)
#in case of paired end we run :
#out <- filterAndTrim(fnFs1, filtFs1,fnRs1,filtRs1,truncLen =(240,160),maxN=0, maxEE=c(2,2), truncQ=2, rm.phix=TRUE, compress=TRUE, multithread=TRUE, minQ=20, verbose=TRUE)
#out3 <- filterAndTrim(fnFs1, filtFs1, fnRs1, filtRs1, truncLen=c(240,160),maxN=0, maxEE=c(2,2), truncQ=2, rm.phix=TRUE,compress=TRUE, multithread=TRUE)
out1
saveRDS(out1, file.path(output_dir, "out1.rds"))
##Generate the Filtered Profiles
#Perform Quality check and generate new profiles for filtered sequences:
#plotQualityProfile(filtFs[1:4])
#plotQualityProfile(filtRs[1:4])

##Error Rate Profiles
#Generate the Estimated Error Rates for Forward and Reverse Sequences
#in case of single end run only:
errF <- learnErrors(filtFs1, multithread=TRUE)
saveRDS(errF, file.path(output_dir, "errF.rds"))

#in case of paired end we run also:
#errR <- learnErrors(filtRs1, multithread=TRUE)

#To visualize the estimated error rates:
#plotErrors(errF, nominalQ=TRUE)

#Third step is the dereplication:
derepFs1 <- derepFastq(filtFs1, verbose=TRUE)
saveRDS(derepFs1, file.path(output_dir, "derepFs1.rds"))
#in case of paired end we run also:
#derepRs <- derepFastq(filtRs, verbose=TRUE)
#derepFs1

##Sample Inference:
#Fourth step is the sample inference:
dadaFs1 <- dada(derepFs1, err=errF, multithread=TRUE)
#in case of paired end we run also:
#dadaRs1 <- dada(derepRs1, err=errR, multithread=TRUE)

#5th step: Merging paired reads: this one is used only in case of paired end sequences
#mergers <- mergePairs(dadaFs, derepFs, verbose=TRUE)
#mergers <- mergePairs(dadaFs, derepFs, dadaRs, derepRs, verbose=TRUE)
#mergers
#6th: Last step is sequence table construction:
seqtab1 <- makeSequenceTable(dadaFs1)
#in case of paired end we use instead of this command this one:
#seqtab <- makeSequenceTable(mergers)
seqtab1
saveRDS(seqtab1, file.path(output_dir, "seqtab1.rds"))
seqtab1<-readRDS("C:/Users/YLS/Documents/IBD1356/rds_outputs_IBD1356/seqtab1.rds")
# View No. of samples and No. of variants
dim(seqtab1)

# View the distribution of the sequences
table(nchar(getSequences(seqtab1)))
#
# Then, removing the chimeras: # This step could take long if verbose=TRUE
data2 <- removeBimeraDenovo(seqtab1, method="consensus", multithread=TRUE, verbose=FALSE)
dim(data2)
# Check the no. of samples and no. of variants when chimeras are removed
#dim(data)
#Finally, Assigning the taxonomy:
taxa2 <- assignTaxonomy(data2, "C:/Users/YLS/Documents/autismMLandDL/silva_nr99_v138.1_train_set.fa.gz", multithread=TRUE)
taxa2 <- addSpecies(taxa2, "C:/Users/YLS/Documents/autismMLandDL/silva_species_assignment_v138.1.fa.gz")
saveRDS(taxa2, file.path(output_dir, "taxa127.rds"))
dim(taxa2)

#save it
write.csv(taxa2,"taxa127.csv")
# To see the first lines of taxonomy:
#unname(head(taxa))


#This part is to add a column indicationg if the samples are for patients or controls, this information is on the metadata


# Read metadata table
metadata2<- runaccession_diagnosis

View(metadata2)


#convert data to dataframe
data2<-data.frame(data2)
View(data2)
row.names(data2)


# convert run accession index column to colum name
df2 <- cbind(run_accession = rownames(data2), data2)
View(df2)

# convert index column to intager index
rownames(df2) <- 1:nrow(df2)
View(df2)
saveRDS(df2, file.path(output_dir, "df2_asv_table.rds"))
View(metadata2)

library("dplyr")
library("rlang")
#install.packages("rlang")
#concatenate metadata and ASV table based on run accesion columncomb<-full_join(USA,df,by="run_accession")
comb1<- full_join(metadata2,df2,by="run_accession",copy=TRUE)
View(comb1)
write.csv(comb1,"data127_joined.csv")
saveRDS(comb1, file.path(output_dir, "combined_data127_metadata_asv.rds"))


