﻿// Copyright (c) Microsoft Corporation. All rights reserved.
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
        }

        internal static async Task<IReadOnlyList<float>> VectorizeAsync(OpenAIClient openAIClient, string text)
        {
            EmbeddingsOptions embeddingsOptions = new(text);
            Embeddings embeddings = await openAIClient.GetEmbeddingsAsync(ModelName, embeddingsOptions);

            // TODO: ToList() is unnecessarily expensive. We need to rationalize the Open AI output and Search input before GA.
            return embeddings.Data[0].Embedding;
        }

        internal static SearchIndex GetHotelIndex(string name)
        {
            string vectorSearchConfigName = "my-vector-config";

            SearchIndex searchIndex = new(name)
            {
                VectorSearch = new()
                {
                    AlgorithmConfigurations =
                    {
                        new VectorSearchAlgorithmConfiguration(vectorSearchConfigName, "hnsw")
                    }
                },
                Fields =
                {
                    new SimpleField("HotelId", SearchFieldDataType.String) { IsKey = true, IsFilterable = true, IsSortable = true, IsFacetable = true },
                    new SearchableField("HotelName") { IsFilterable = true, IsSortable = true },
                    new SearchableField("Description") { IsFilterable = true },
                    new SearchField("DescriptionVector", SearchFieldDataType.Collection(SearchFieldDataType.Single))
                    {
                        IsSearchable = true,
                        Dimensions = ModelDimensions,
                        VectorSearchConfiguration = vectorSearchConfigName
                    },
                    new SearchableField("Category") { IsFilterable = true, IsSortable = true, IsFacetable = true }
                }
            };

            return searchIndex;
        }

        internal static async Task SingleVectorSearch(SearchClient client, OpenAIClient openAIClient)
        {
            var vectorizedResult = await VectorizeAsync(openAIClient, "Top hotels in town");
            Assert.NotNull(vectorizedResult);
            Assert.GreaterOrEqual(vectorizedResult.Count, 1);
            await Task.Delay(TimeSpan.FromSeconds(1));

            var vector = new SearchQueryVector { Value = vectorizedResult, K = 3, Fields = "DescriptionVector" };
            SearchResults<Hotel> response = await client.SearchAsync<Hotel>(
                   null,
                   new SearchOptions
                   {
                       Vector = vector,
                       Select = { "HotelId", "HotelName" }
                   });

            int count = 0;
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

            var vector = new SearchQueryVector { Value = vectorizedResult, K = 3, Fields = "DescriptionVector" };
            SearchResults<Hotel> response = await client.SearchAsync<Hotel>(
                    null,
                    new SearchOptions
                    {
                        Vector = vector,
                        Filter = "Category eq 'Luxury'",
                        Select = { "HotelId", "HotelName", "Category" }
                    });

            int count = 0;
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

            var vector = new SearchQueryVector { Value = vectorizedResult, K = 3, Fields = "DescriptionVector" };
            SearchResults<Hotel> response = await client.SearchAsync<Hotel>(
                    "Top hotels in town",
                    new SearchOptions
                    {
                        Vector = vector,
                        Select = { "HotelId", "HotelName" },
                    });

            int count = 0;
            await foreach (SearchResult<Hotel> result in response.GetResultsAsync())
            {
                count++;
                Hotel doc = result.Document;
                Console.WriteLine($"{doc.HotelId}: {doc.HotelName}");
            }
            Assert.AreEqual(4, count); // HotelId - 3, 1, 5, 2
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

        /// <summary>
        /// Get Sample documents.
        /// </summary>
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
