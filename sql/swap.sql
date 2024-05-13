SELECT  a.*
        ,b.country
        ,b.language
        ,b.device_create_date
        ,b.imei_level_1_user_source
        ,b.imei_level_2_user_source
        ,b.imei_level_3_user_source
        ,b.lifecycle_tag
FROM    (
            SELECT  identity
                    ,concat_ws('-', from_coin, to_coin) coin_pair
                    ,concat_ws('-', from_chain, to_chain) chain_pair
                    ,market
                    ,from_amt_usd
                    ,IF(date(create_time) = date_sub(current_date(), 2),0,1) target
                    ,date(create_time) dt
                    ,order_no
                    ,hour(create_time) create_hour
            FROM    bg_dw.dwd_bk_trs_swap_order_df
            WHERE   pt = '${bizdate}'
            AND     order_status = 4
            AND     date(create_time) BETWEEN date_sub(current_date(), 2) AND date_sub(current_date(), 1)
        ) a
LEFT JOIN   (
                SELECT  identity
                        ,country
                        ,language
                        ,date(device_create_time) device_create_date
                        ,imei_level_1_user_source
                        ,imei_level_2_user_source
                        ,imei_level_3_user_source
                        ,lifecycle_tag
                FROM    bg_dw.dws_bk_usr_profile_subset_df
                WHERE   pt = '${bizdate}'
            ) b
ON      a.identity = b.identity
;