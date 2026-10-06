from qdrant_client import AsyncQdrantClient, QdrantClient, models


def first_filter(user_uuid):

    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(
                    value=str(user_uuid),
                ),
            ),
            models.FieldCondition(
                key="type",
                match=models.MatchValue(
                    value="doc",
                ),
            ),
        ]
    )


def second_filter(user_uuid, docs_name_md5):

    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(
                    value=str(user_uuid),
                ),
            ),
            models.FieldCondition(
                key="type",
                match=models.MatchValue(
                    value="doc",
                ),
            ),
        ],
        should=[
            models.FieldCondition(
                key="file_md5",
                match=models.MatchValue(
                    value=doc[0],
                ),
            )
            for doc in docs_name_md5
        ],
    )


def last_filter(item, user_uuid, md5):
    l_idx_filter = []
    for l_idx in item:
        if l_idx[0]:
            l_idx_filter.append(
                models.Filter(
                    must=[
                        # models.FieldCondition(
                        #     key="l_chunk_idx_from",
                        #     range=models.Range(
                        #         gt=None,
                        #         gte=None,
                        #         lt=None,
                        #         lte=l_idx[1],
                        #     ),
                        # ),
                        # models.FieldCondition(
                        #     key="l_chunk_idx_to",
                        #     range=models.Range(
                        #         gt=None,
                        #         gte=l_idx[0],
                        #         lt=None,
                        #         lte=None,
                        #     ),
                        # ),
                        models.FieldCondition(
                            key="idx",
                            range=models.Range(
                                gt=None,
                                gte=l_idx[0],
                                lt=None,
                                lte=l_idx[1],
                            ),
                        ),
                    ],
                )
            )
        else:
            l_idx_filter.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="idx",
                            range=models.Range(
                                gt=None,
                                gte=l_idx[1] - 1,
                                lt=None,
                                lte=l_idx[1] + 1,
                            ),
                        ),
                    ],
                )
            )

    filter_3 = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=str(user_uuid)),
            ),
            models.FieldCondition(
                key="file_md5",
                match=models.MatchValue(value=md5),
            ),
            models.FieldCondition(
                key="type",
                match=models.MatchValue(
                    value="doc",
                ),
            ),
        ],
        should=l_idx_filter,
    )

    return filter_3
