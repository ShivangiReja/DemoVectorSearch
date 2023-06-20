// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Threading.Tasks;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.AI.OpenAI;
using System.Collections.Generic;
using System.Linq;
using Azure.Search.Documents.Models;
using NUnit.Framework;

namespace Azure.Search.Documents.Tests.Samples
{
    public partial class Program
    {
        private const string ModelName = "text-embedding-ada-002";
        private const int ModelDimensions = 1536;
        private const string SemanticSearchConfigName = "my-semantic-config";

        public static async Task Main(string[] args)
        {
            Uri endpoint;
            AzureKeyCredential credential;

            //-----------Create Index---------------------
            endpoint = new(Environment.GetEnvironmentVariable("SEARCH_ENDPOINT"));
            credential = new(Environment.GetEnvironmentVariable("SEARCH_ADMIN_API_KEY"));
            SearchIndexClient indexClient = new(endpoint, credential);

            var indexName = "vectorsearchindex";
            SearchIndex index = GetHotelIndex(indexName);
            await indexClient.CreateIndexAsync(index);

            //--------Upload data------------------------
            SearchClient searchClient = new(endpoint, indexName, credential);

            endpoint = new(Environment.GetEnvironmentVariable("OpenAIEndpoint"));
            credential = new(Environment.GetEnvironmentVariable("OpenAIKey"));
            OpenAIClient openAIClient = new(endpoint, credential);

            Hotel[] hotelDocuments = await GetHotelDocumentsAsync(openAIClient);
            await searchClient.IndexDocumentsAsync(IndexDocumentsBatch.Upload(hotelDocuments));

            //-------Single Vector Search-------------------
            await SingleVectorSearch(searchClient, openAIClient);

            //--------Single Vector Search with filter-------------
            await SingleVectorSearchWithFilter(searchClient, openAIClient);

            //-------Simple Hybrid Search-------------------
            await SimpleHybridSearch(searchClient, openAIClient);

            //-------Semantic Hybrid Search-------------------
            await SemanticHybridSearch(searchClient, openAIClient);
        }

        internal static async Task SingleVectorSearch(SearchClient client, OpenAIClient openAIClient)
        {
            var vectorizedResult = await VectorizeAsync(openAIClient, "Top hotels in town");
            Assert.NotNull(vectorizedResult);
            Assert.GreaterOrEqual(vectorizedResult.Count, 1);
            await Task.Delay(TimeSpan.FromSeconds(1));

            var vector = new SearchQueryVector { Value = vectorizedResult, KNearestNeighborsCount = 3, Fields = "DescriptionVector" };
            SearchResults<Hotel> response = await client.SearchAsync<Hotel>(
                   null,
                   new SearchOptions
                   {
                       Vector = vector,
                       Select = { "HotelId", "HotelName" }
                   });

            int count = 0;
            Console.WriteLine($"\nSingle Vector Search Results:");
            await foreach (SearchResult<Hotel> result in response.GetResultsAsync())
            {
                count++;
                Hotel doc = result.Document;
                Console.WriteLine($"{doc.HotelId}: {doc.HotelName}");
            }
            Assert.AreEqual(3, count); // HotelId - 3, 1, 5
        }

        internal static async Task SingleVectorSearchWithFilter(SearchClient client, OpenAIClient openAIClient)
        {
            var vectorizedResult = await VectorizeAsync(openAIClient, "Top hotels in town");
            Assert.NotNull(vectorizedResult);
            Assert.GreaterOrEqual(vectorizedResult.Count, 1);

            var vector = new SearchQueryVector { Value = vectorizedResult, KNearestNeighborsCount = 3, Fields = "DescriptionVector" };
            SearchResults<Hotel> response = await client.SearchAsync<Hotel>(
                    null,
                    new SearchOptions
                    {
                        Vector = vector,
                        Filter = "Category eq 'Luxury'",
                        Select = { "HotelId", "HotelName", "Category" }
                    });

            int count = 0;
            Console.WriteLine($"\nSingle Vector Search With Filter Results:");
            await foreach (SearchResult<Hotel> result in response.GetResultsAsync())
            {
                count++;
                Hotel doc = result.Document;
                Console.WriteLine($"{doc.HotelId}: {doc.HotelName}");
            }
            Assert.AreEqual(2, count); // HotelId - 1, 4
        }

        internal static async Task SimpleHybridSearch(SearchClient client, OpenAIClient openAIClient)
        {
            var vectorizedResult = await VectorizeAsync(openAIClient, "Top hotels in town");
            Assert.NotNull(vectorizedResult);
            Assert.GreaterOrEqual(vectorizedResult.Count, 1);

            var vector = new SearchQueryVector { Value = vectorizedResult, KNearestNeighborsCount = 3, Fields = "DescriptionVector" };
            SearchResults<Hotel> response = await client.SearchAsync<Hotel>(
                    "Top hotels in town",
                    new SearchOptions
                    {
                        Vector = vector,
                        Select = { "HotelId", "HotelName" },
                    });

            int count = 0;
            Console.WriteLine($"\nSimple Hybrid Search Results:");
            await foreach (SearchResult<Hotel> result in response.GetResultsAsync())
            {
                count++;
                Hotel doc = result.Document;
                Console.WriteLine($"{doc.HotelId}: {doc.HotelName}");
            }
            Assert.AreEqual(4, count); // HotelId - 3, 1, 5, 2
        }

        internal static async Task SemanticHybridSearch(SearchClient client, OpenAIClient openAIClient)
        {
            var vectorizedResult = await VectorizeAsync(openAIClient, "Top hotels in town");
            Assert.NotNull(vectorizedResult);
            Assert.GreaterOrEqual(vectorizedResult.Count, 1);

            var vector = new SearchQueryVector { Value = vectorizedResult, KNearestNeighborsCount = 3, Fields = "DescriptionVector" };
            SearchResults<Hotel> response = await client.SearchAsync<Hotel>(
                    "Is there any luxury hotel in town?",
                    new SearchOptions
                    {
                        Vector = vector,
                        Select = { "HotelId", "HotelName", "Description", "Category" },
                        QueryType = SearchQueryType.Semantic,
                        QueryLanguage = QueryLanguage.EnUs,
                        SemanticConfigurationName = SemanticSearchConfigName,
                        QueryCaption = QueryCaptionType.Extractive,
                        QueryAnswer = QueryAnswerType.Extractive,
                        QueryCaptionHighlightEnabled = true
                    });

            int count = 0;
            Console.WriteLine($"\nSemantic Hybrid Search Results:");

            Console.WriteLine($"\nQuery Answer:");
            foreach (AnswerResult result in response.Answers)
            {
                Console.WriteLine($"Answer Highlights: {result.Highlights}");
                Console.WriteLine($"Answer Text: {result.Text}");
            }

            await foreach (SearchResult<Hotel> result in response.GetResultsAsync())
            {
                count++;
                Hotel doc = result.Document;
                Console.WriteLine($"\nHotelId: {doc.HotelId} \n HotelName: {doc.HotelName} \n Category: {doc.Category} \n Description: {doc.Description}");

                if (result.Captions != null)
                {
                    var caption = result.Captions.FirstOrDefault();
                    if (caption.Highlights != null && caption.Highlights != "")
                    {
                        Console.WriteLine($"Caption Highlights: {caption.Highlights}");
                    }
                    else
                    {
                        Console.WriteLine($"Caption Text: {caption.Text}");
                    }
                }
            }

            Assert.AreEqual(5, count); // HotelId - 1, 5, 3, 4, 2
        }

        /// <summary> Get a <see cref="SearchIndex"/> for the Hotels sample data. </summary>
        internal static SearchIndex GetHotelIndex(string name)
        {
            string vectorSearchConfigName = "my-vector-config";

            SearchIndex searchIndex = new(name)
            {
                Fields =
                {
                    new SimpleField("HotelId", SearchFieldDataType.String) { IsKey = true, IsFilterable = true, IsSortable = true, IsFacetable = true },
                    new SearchableField("HotelName") { IsFilterable = true, IsSortable = true },
                    new SearchableField("Description") { IsFilterable = true },
                    new SearchField("DescriptionVector", SearchFieldDataType.Collection(SearchFieldDataType.Single))
                    {
                        IsSearchable = true,
                        VectorSearchDimensions = ModelDimensions,
                        VectorSearchConfiguration = vectorSearchConfigName
                    },
                    new SearchableField("Category") { IsFilterable = true, IsSortable = true, IsFacetable = true }
                },
                VectorSearch = new()
                {
                    AlgorithmConfigurations =
                    {
                        new HnswVectorSearchAlgorithmConfiguration(vectorSearchConfigName)
                    }
                },
                SemanticSettings = new()
                {
                    Configurations =
                    {
                       new SemanticConfiguration(SemanticSearchConfigName, new()
                       {
                           TitleField = new(){ FieldName = "HotelName" },
                           ContentFields =
                           {
                               new() { FieldName = "Description" }
                           },
                           KeywordFields =
                           {
                               new() { FieldName = "Category" }
                           }

                       })
                    }
                },
            };

            return searchIndex;
        }

        internal static async Task<IReadOnlyList<float>> VectorizeAsync(OpenAIClient openAIClient, string text)
        {
            EmbeddingsOptions embeddingsOptions = new(text);
            Embeddings embeddings = await openAIClient.GetEmbeddingsAsync(ModelName, embeddingsOptions);

            return embeddings.Data[0].Embedding;
        }

        internal class Hotel
        {
            public string HotelId { get; set; }
            public string HotelName { get; set; }
            public string Description { get; set; }
            public IReadOnlyList<float> DescriptionVector { get; set; }
            public string Category { get; set; }

            public override bool Equals(object obj) =>
                obj is Hotel other &&
                HotelId == other.HotelId &&
                HotelName == other.HotelName &&
                Description == other.Description &&
                DescriptionVector == other.DescriptionVector &&
                Category == other.Category;

            public override int GetHashCode() => HotelId?.GetHashCode() ?? 0;
        }

        /// <summary> Get Sample documents. </summary>
        internal static async Task<Hotel[]> GetHotelDocumentsAsync(OpenAIClient openAIClient)
        {
            return new[]
            {
                new Hotel()
                {
                    HotelId = "1",
                    HotelName = "Fancy Stay",
                    Description = "Best hotel in town if you like luxury hotels.",
                    DescriptionVector = await VectorizeAsync(openAIClient, "Best hotel in town if you like luxury hotels."),
                    Category = "Luxury",
                },
                new Hotel()
                {
                    HotelId = "2",
                    HotelName = "Roach Motel",
                    Description = "Cheapest hotel in town. Infact, a motel.",
                    DescriptionVector = await VectorizeAsync(openAIClient, "Cheapest hotel in town. Infact, a motel."),
                    Category = "Budget",
                },
                new Hotel()
                {
                    HotelId = "3",
                    HotelName = "EconoStay",
                    Description = "Very popular hotel in town.",
                    DescriptionVector = await VectorizeAsync(openAIClient, "Very popular hotel in town."),
                    Category = "Budget",
                },
                new Hotel()
                {
                    HotelId = "4",
                    HotelName = "Modern Stay",
                    Description = "Modern architecture, very polite staff and very clean. Also very affordable.",
                    DescriptionVector = await VectorizeAsync(openAIClient, "Modern architecture, very polite staff and very clean. Also very affordable."),
                    Category = "Luxury",
                },
                new Hotel()
                {
                    HotelId = "5",
                    HotelName = "Secret Point",
                    Description = "One of the best hotel in town. The hotel is ideally located on the main commercial artery of the city in the heart of New York.",
                    DescriptionVector = await VectorizeAsync(openAIClient, "One of the best hotel in town. The hotel is ideally located on the main commercial artery of the city in the heart of New York."),
                    Category = "Boutique",
                }
            };
        }
    }
}
