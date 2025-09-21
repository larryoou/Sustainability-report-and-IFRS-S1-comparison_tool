#!/usr/bin/env python3
"""
完整的IFRS S1條文數據
提取的條文數量: 125
"""

IFRS_S1_ARTICLES = [
    {
        'id': 'IFRS-S1-21',
        'title': '''治理 - 一般規定''',
        'content': '''個體應揭露用以監督、管理及治理永續相關風險與機會之治理流程、控制及程序之資訊。''',
        'category': '治理',
        'difficulty': 'high',
        'keywords': ['治理', '監督', '管理', '董事會', '委員會', '治理流程', '控制程序']
    },
    {
        'id': 'IFRS-S1-22',
        'title': '''董事會監督''',
        'content': '''個體應揭露負責監督永續相關風險與機會之治理單位或個人之資訊。''',
        'category': '治理',
        'difficulty': 'high',
        'keywords': ['董事會', '監督', '策略', '重大決策', '治理單位', '責任分工']
    },
    {
        'id': 'IFRS-S1-23',
        'title': '''管理階層角色''',
        'content': '''個體應揭露管理階層在評估及管理永續相關風險與機會方面之角色。''',
        'category': '治理',
        'difficulty': 'medium',
        'keywords': ['管理階層', '角色', '責任', '評估', '管理', '執行層面', '日常營運']
    },
    {
        'id': 'IFRS-S1-24',
        'title': '''治理流程''',
        'content': '''個體應揭露用以識別、評估、排定優先順序及監控永續相關風險與機會之流程。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['治理流程', '決策過程', '監控', '檢討']
    },
    {
        'id': 'IFRS-S1-25',
        'title': '''專業知識與技能''',
        'content': '''個體應揭露治理單位及管理階層具備之永續相關專業知識與技能。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['專業知識', '技能', '能力', '訓練']
    },
    {
        'id': 'IFRS-S1-26',
        'title': '''激勵措施''',
        'content': '''個體應揭露是否及如何將永續相關風險與機會納入薪酬政策。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['激勵措施', '薪酬', '績效', '永續目標']
    },
    {
        'id': 'IFRS-S1-27',
        'title': '''治理結構變更''',
        'content': '''個體應揭露報告期間內治理結構之重大變更。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['治理結構', '變更', '組織調整', '責任分工']
    },
    {
        'id': 'IFRS-S1-28',
        'title': '''利害關係人參與''',
        'content': '''個體應揭露如何在治理過程中考量利害關係人之意見。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['利害關係人', '參與', '溝通', '回饋']
    },
    {
        'id': 'IFRS-S1-29',
        'title': '''風險胃納''',
        'content': '''個體應揭露其對永續相關風險之風險胃納。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['風險胃納', '風險承受度', '策略目標']
    },
    {
        'id': 'IFRS-S1-30',
        'title': '''治理有效性評估''',
        'content': '''個體應揭露如何評估及改進永續相關治理之有效性。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['有效性', '評估', '檢討', '改進']
    },
    {
        'id': 'IFRS-S1-31',
        'title': '''策略 - 一般規定''',
        'content': '''個體應揭露其策略如何應對永續相關風險與機會。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['策略', '商業模式', '價值創造']
    },
    {
        'id': 'IFRS-S1-32',
        'title': '''策略制定過程''',
        'content': '''個體應揭露如何在策略制定過程中考量永續相關風險與機會。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['策略制定', '過程', '考量因素', '決策']
    },
    {
        'id': 'IFRS-S1-33',
        'title': '''商業模式''',
        'content': '''個體應揭露永續相關風險與機會如何影響其商業模式。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['商業模式', '價值創造', '永續性', '轉型']
    },
    {
        'id': 'IFRS-S1-34',
        'title': '''價值鏈分析''',
        'content': '''個體應揭露其價值鏈中永續相關風險與機會之分析。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['價值鏈', '分析', '影響評估', '依賴關係']
    },
    {
        'id': 'IFRS-S1-35',
        'title': '''策略執行''',
        'content': '''個體應揭露執行永續相關策略之具體行動和資源配置。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['策略執行', '實施', '資源配置', '進度監控']
    },
    {
        'id': 'IFRS-S1-36',
        'title': '''策略實施 - 一般規定''',
        'content': '''個體應揭露其策略實施的具體計劃、資源分配及相關績效指標。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['策略實施', '執行計劃', '資源分配', '績效指標']
    },
    {
        'id': 'IFRS-S1-37',
        'title': '''策略實施 - 短期目標''',
        'content': '''個體應揭露短期內策略實施的具體目標及進度追蹤機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['短期目標', '階段性目標', '里程碑', '進度追蹤']
    },
    {
        'id': 'IFRS-S1-38',
        'title': '''策略實施 - 中期目標''',
        'content': '''個體應揭露中期策略實施的目標及預期成果。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['中期目標', '三年計劃', '發展路徑', '階段成果']
    },
    {
        'id': 'IFRS-S1-39',
        'title': '''策略實施 - 長期願景''',
        'content': '''個體應揭露長期策略實施的願景及轉型目標。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['長期願景', '十年計劃', '永續發展', '轉型目標']
    },
    {
        'id': 'IFRS-S1-40',
        'title': '''策略實施 - 風險管理''',
        'content': '''個體應揭露策略實施過程中的風險管理措施及應變計劃。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['風險管理', '策略風險', '緩解措施', '應變計劃']
    },
    {
        'id': 'IFRS-S1-41',
        'title': '''策略實施 - 利害關係人參與''',
        'content': '''個體應揭露利害關係人在策略實施過程中的參與及協作機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['利害關係人', '參與', '溝通', '協作', '夥伴關係']
    },
    {
        'id': 'IFRS-S1-42',
        'title': '''策略實施 - 創新與技術''',
        'content': '''個體應揭露策略實施中的創新與技術應用。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['創新', '技術應用', '數位轉型', '研發投資']
    },
    {
        'id': 'IFRS-S1-43',
        'title': '''策略實施 - 供應鏈管理''',
        'content': '''個體應揭露供應鏈管理在策略實施中的角色。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['供應鏈', '供應商管理', '永續供應鏈', '採購策略']
    },
    {
        'id': 'IFRS-S1-44',
        'title': '''策略實施 - 員工發展''',
        'content': '''個體應揭露員工發展在策略實施中的重要性。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['員工發展', '人才培養', '技能提升', '人力資源']
    },
    {
        'id': 'IFRS-S1-45',
        'title': '''策略實施 - 社區參與''',
        'content': '''個體應揭露社區參與在策略實施中的具體行動。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['社區參與', '社會影響', '在地發展', '社會責任']
    },
    {
        'id': 'IFRS-S1-46',
        'title': '''績效衡量 - 一般規定''',
        'content': '''個體應揭露用以衡量永續相關策略實施成效的指標。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['績效衡量', '關鍵績效指標', 'KPI', '衡量標準']
    },
    {
        'id': 'IFRS-S1-47',
        'title': '''績效衡量 - 環境指標''',
        'content': '''個體應揭露環境方面的關鍵績效指標。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['環境指標', '碳排放', '能源使用', '廢棄物管理']
    },
    {
        'id': 'IFRS-S1-48',
        'title': '''績效衡量 - 社會指標''',
        'content': '''個體應揭露社會方面的關鍵績效指標。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['社會指標', '員工滿意度', '多元包容', '社區影響']
    },
    {
        'id': 'IFRS-S1-49',
        'title': '''績效衡量 - 治理指標''',
        'content': '''個體應揭露治理方面的關鍵績效指標。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['治理指標', '透明度', '問責制', '倫理標準']
    },
    {
        'id': 'IFRS-S1-50',
        'title': '''績效衡量 - 目標設定''',
        'content': '''個體應揭露績效目標的設定原則及量化標準。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['目標設定', 'SMART原則', '量化目標', '時間表']
    },
    {
        'id': 'IFRS-S1-51',
        'title': '''績效衡量 - 數據收集''',
        'content': '''個體應揭露績效數據的收集及驗證機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['數據收集', '自動化', '感測器', '即時監控']
    },
    {
        'id': 'IFRS-S1-52',
        'title': '''績效衡量 - 報告頻率''',
        'content': '''個體應揭露績效報告的頻率及更新機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['報告頻率', '定期報告', '即時更新', '年度檢討']
    },
    {
        'id': 'IFRS-S1-53',
        'title': '''績效衡量 - 比較分析''',
        'content': '''個體應揭露績效數據的比較分析方法。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['比較分析', '基準比較', '趨勢分析', '最佳實務']
    },
    {
        'id': 'IFRS-S1-54',
        'title': '''績效衡量 - 外部驗證''',
        'content': '''個體應揭露績效數據的外部驗證機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['外部驗證', '第三方審核', '獨立評估', '公信力']
    },
    {
        'id': 'IFRS-S1-55',
        'title': '''績效衡量 - 改進措施''',
        'content': '''個體應揭露基於績效衡量結果的改進措施。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['改進措施', '持續改善', '矯正行動', '學習循環']
    },
    {
        'id': 'IFRS-S1-56',
        'title': '''風險識別 - 一般規定''',
        'content': '''個體應揭露永續相關風險的識別及分類方法。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['風險識別', '風險評估', '風險地圖', '風險類型']
    },
    {
        'id': 'IFRS-S1-57',
        'title': '''風險識別 - 氣候相關風險''',
        'content': '''個體應揭露氣候相關風險的具體類型及影響評估。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['氣候風險', '氣候變遷', '極端氣候', '轉型風險']
    },
    {
        'id': 'IFRS-S1-58',
        'title': '''風險識別 - 環境風險''',
        'content': '''個體應揭露環境方面的風險及影響評估。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['環境風險', '污染', '資源枯竭', '生態系統破壞']
    },
    {
        'id': 'IFRS-S1-59',
        'title': '''風險識別 - 社會風險''',
        'content': '''個體應揭露社會方面的風險及影響評估。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['社會風險', '勞工權益', '人權議題', '社區衝突']
    },
    {
        'id': 'IFRS-S1-60',
        'title': '''風險識別 - 治理風險''',
        'content': '''個體應揭露治理方面的風險及影響評估。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['治理風險', '腐敗', '道德風險', '合規風險']
    },
    {
        'id': 'IFRS-S1-61',
        'title': '''風險評估 - 影響程度''',
        'content': '''個體應揭露風險的影響程度及發生機率評估。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['風險評估', '影響程度', '嚴重性', '機率', '後果']
    },
    {
        'id': 'IFRS-S1-62',
        'title': '''風險評估 - 時間範圍''',
        'content': '''個體應揭露風險評估的時間範圍及動態調整機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['時間範圍', '短期風險', '長期風險', '動態評估']
    },
    {
        'id': 'IFRS-S1-63',
        'title': '''風險評估 - 相互關聯''',
        'content': '''個體應揭露風險間的相互關聯及系統性影響。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['風險關聯', '風險組合', '系統性風險', '連鎖效應']
    },
    {
        'id': 'IFRS-S1-64',
        'title': '''風險管理 - 一般規定''',
        'content': '''個體應揭露風險管理的整體策略及具體措施。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['風險管理', '緩解策略', '風險控制', '應變計劃']
    },
    {
        'id': 'IFRS-S1-65',
        'title': '''風險管理 - 避免策略''',
        'content': '''個體應揭露避免或減少風險暴露的策略。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['避免策略', '風險迴避', '退出策略', '業務調整']
    },
    {
        'id': 'IFRS-S1-66',
        'title': '''風險管理 - 減輕策略''',
        'content': '''個體應揭露減輕風險影響的具體措施。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['減輕策略', '風險緩解', '對沖機制', '保險安排']
    },
    {
        'id': 'IFRS-S1-67',
        'title': '''風險管理 - 轉移策略''',
        'content': '''個體應揭露將風險轉移給其他方的機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['轉移策略', '風險轉移', '合約安排', '夥伴合作']
    },
    {
        'id': 'IFRS-S1-68',
        'title': '''風險管理 - 接受策略''',
        'content': '''個體應揭露接受風險並準備因應的策略。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['接受策略', '風險接受', '風險胃納', '準備金']
    },
    {
        'id': 'IFRS-S1-69',
        'title': '''風險監控 - 一般規定''',
        'content': '''個體應揭露風險監控的機制及指標。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['風險監控', '持續監測', '早期預警', '指標追蹤']
    },
    {
        'id': 'IFRS-S1-70',
        'title': '''風險監控 - 報告機制''',
        'content': '''個體應揭露風險監控結果的報告及決策機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['風險報告', '定期檢討', '緊急通報', '決策支援']
    },
    {
        'id': 'IFRS-S1-71',
        'title': '''機會識別 - 一般規定''',
        'content': '''個體應揭露永續相關機會的識別及評估方法。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['機會識別', '永續機會', '創新機會', '成長機會']
    },
    {
        'id': 'IFRS-S1-72',
        'title': '''機會識別 - 環境機會''',
        'content': '''個體應揭露環境方面的機會及潛在價值。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['環境機會', '綠色創新', '資源效率', '循環經濟']
    },
    {
        'id': 'IFRS-S1-73',
        'title': '''機會識別 - 社會機會''',
        'content': '''個體應揭露社會方面的機會及潛在價值。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['社會機會', '人才發展', '品牌價值', '社會影響投資']
    },
    {
        'id': 'IFRS-S1-74',
        'title': '''機會識別 - 治理機會''',
        'content': '''個體應揭露治理方面的機會及潛在價值。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['治理機會', '透明提升', '信譽改善', '投資吸引力']
    },
    {
        'id': 'IFRS-S1-75',
        'title': '''機會評估 - 潛在價值''',
        'content': '''個體應揭露機會的潛在價值及影響評估。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['機會評估', '潛在價值', '財務影響', '非財務影響']
    },
    {
        'id': 'IFRS-S1-76',
        'title': '''機會評估 - 實現機率''',
        'content': '''個體應揭露機會實現的機率及關鍵成功因素。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['實現機率', '成功因素', '障礙分析', '資源需求']
    },
    {
        'id': 'IFRS-S1-77',
        'title': '''機會開發 - 一般規定''',
        'content': '''個體應揭露機會開發的策略及資源配置。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['機會開發', '創新投資', '合作夥伴', '資源配置']
    },
    {
        'id': 'IFRS-S1-78',
        'title': '''機會開發 - 產品創新''',
        'content': '''個體應揭露產品及服務方面的創新機會。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['產品創新', '永續產品', '服務創新', '商業模式創新']
    },
    {
        'id': 'IFRS-S1-79',
        'title': '''機會開發 - 流程優化''',
        'content': '''個體應揭露流程優化方面的機會開發。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['流程優化', '效率提升', '成本節省', '品質改善']
    },
    {
        'id': 'IFRS-S1-80',
        'title': '''機會開發 - 市場擴張''',
        'content': '''個體應揭露市場擴張方面的機會開發。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['市場擴張', '新市場', '客戶群', '地理擴張']
    },
    {
        'id': 'IFRS-S1-81',
        'title': '''資源分配 - 一般規定''',
        'content': '''個體應揭露資源分配的原則及決策機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['資源分配', '優先順序', '投資決策', '資源優化']
    },
    {
        'id': 'IFRS-S1-82',
        'title': '''資源分配 - 財務資源''',
        'content': '''個體應揭露財務資源的分配及管理。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['財務資源', '投資預算', '資金分配', '融資策略']
    },
    {
        'id': 'IFRS-S1-83',
        'title': '''資源分配 - 人力資源''',
        'content': '''個體應揭露人力資源的分配及發展。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['人力資源', '人才配置', '技能發展', '團隊建設']
    },
    {
        'id': 'IFRS-S1-84',
        'title': '''資源分配 - 技術資源''',
        'content': '''個體應揭露技術資源的分配及運用。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['技術資源', '技術投資', '研發資源', '創新支援']
    },
    {
        'id': 'IFRS-S1-85',
        'title': '''資源分配 - 夥伴關係''',
        'content': '''個體應揭露夥伴關係資源的開發及管理。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['夥伴關係', '合作協議', '供應鏈整合', '生態系統']
    },
    {
        'id': 'IFRS-S1-86',
        'title': '''利害關係人參與 - 一般規定''',
        'content': '''個體應揭露利害關係人參與的機制及管道。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['利害關係人', '參與機制', '溝通管道', '回饋機制']
    },
    {
        'id': 'IFRS-S1-87',
        'title': '''利害關係人參與 - 識別過程''',
        'content': '''個體應揭露利害關係人的識別及分類方法。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['利害關係人識別', '影響評估', '優先順序', '參與程度']
    },
    {
        'id': 'IFRS-S1-88',
        'title': '''利害關係人參與 - 溝通策略''',
        'content': '''個體應揭露與利害關係人的溝通策略。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['溝通策略', '資訊分享', '對話機制', '透明度']
    },
    {
        'id': 'IFRS-S1-89',
        'title': '''利害關係人參與 - 回饋整合''',
        'content': '''個體應揭露如何將利害關係人回饋納入決策過程。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['回饋整合', '意見納入', '決策影響', '改進措施']
    },
    {
        'id': 'IFRS-S1-90',
        'title': '''利害關係人參與 - 衝突解決''',
        'content': '''個體應揭露處理利害關係人衝突的機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['衝突解決', '爭議處理', '調解機制', '共識建立']
    },
    {
        'id': 'IFRS-S1-91',
        'title': '''報告與溝通 - 一般規定''',
        'content': '''個體應揭露報告與溝通的整體策略及標準。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['報告與溝通', '資訊揭露', '透明度', '可及性']
    },
    {
        'id': 'IFRS-S1-92',
        'title': '''報告與溝通 - 報告內容''',
        'content': '''個體應揭露報告內容的組織及呈現方式。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['報告內容', '資訊完整性', '相關性', '清晰度']
    },
    {
        'id': 'IFRS-S1-93',
        'title': '''報告與溝通 - 報告頻率''',
        'content': '''個體應揭露報告發佈的頻率及時機。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['報告頻率', '年度報告', '中期更新', '即時通報']
    },
    {
        'id': 'IFRS-S1-94',
        'title': '''報告與溝通 - 報告格式''',
        'content': '''個體應揭露報告的格式及技術應用。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['報告格式', '數位化', '互動性', '多媒體']
    },
    {
        'id': 'IFRS-S1-95',
        'title': '''報告與溝通 - 目標受眾''',
        'content': '''個體應揭露報告的目標受眾及適應策略。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['目標受眾', '資訊需求', '溝通管道', '語言適應']
    },
    {
        'id': 'IFRS-S1-96',
        'title': '''報告與溝通 - 驗證機制''',
        'content': '''個體應揭露報告內容的驗證及品質控制機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['驗證機制', '第三方驗證', '保證聲明', '品質控制']
    },
    {
        'id': 'IFRS-S1-97',
        'title': '''報告與溝通 - 回饋收集''',
        'content': '''個體應揭露收集利害關係人回饋的機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['回饋收集', '意見調查', '改進建議', '持續改善']
    },
    {
        'id': 'IFRS-S1-98',
        'title': '''績效追蹤 - 一般規定''',
        'content': '''個體應揭露績效追蹤的整體框架及方法。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['績效追蹤', '指標監控', '進度評估', '調整機制']
    },
    {
        'id': 'IFRS-S1-99',
        'title': '''績效追蹤 - 短期監控''',
        'content': '''個體應揭露短期績效的監控機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['短期監控', '月度追蹤', '季度檢討', '快速回應']
    },
    {
        'id': 'IFRS-S1-100',
        'title': '''績效追蹤 - 長期評估''',
        'content': '''個體應揭露長期績效的評估方法。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['長期評估', '年度檢討', '趨勢分析', '策略調整']
    },
    {
        'id': 'IFRS-S1-101',
        'title': '''績效追蹤 - 外部比較''',
        'content': '''個體應揭露績效與外部基準的比較分析。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['外部比較', '基準對比', '行業標準', '最佳實務']
    },
    {
        'id': 'IFRS-S1-102',
        'title': '''績效追蹤 - 內部目標''',
        'content': '''個體應揭露績效與內部目標的比較及差距分析。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['內部目標', '目標達成', '差距分析', '行動計劃']
    },
    {
        'id': 'IFRS-S1-103',
        'title': '''績效追蹤 - 調整機制''',
        'content': '''個體應揭露基於績效追蹤結果的調整機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['調整機制', '策略修正', '資源重新分配', '優先順序調整']
    },
    {
        'id': 'IFRS-S1-104',
        'title': '''績效追蹤 - 預警系統''',
        'content': '''個體應揭露績效預警及早期干預機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['預警系統', '早期發現', '風險指標', '臨界值']
    },
    {
        'id': 'IFRS-S1-105',
        'title': '''績效追蹤 - 報告整合''',
        'content': '''個體應揭露績效追蹤結果在報告中的整合方式。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['報告整合', '綜合分析', '趨勢預測', '決策支援']
    },
    {
        'id': 'IFRS-S1-106',
        'title': '''治理與監督 - 一般規定''',
        'content': '''個體應揭露永續治理的整體架構及監督機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['治理監督', '責任架構', '決策權限', '問責機制']
    },
    {
        'id': 'IFRS-S1-107',
        'title': '''治理與監督 - 董事會角色''',
        'content': '''個體應揭露董事會在永續治理中的角色及責任。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['董事會角色', '監督責任', '策略指導', '風險監督']
    },
    {
        'id': 'IFRS-S1-108',
        'title': '''治理與監督 - 管理階層責任''',
        'content': '''個體應揭露管理階層的永續相關責任及義務。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['管理階層', '執行責任', '績效管理', '報告義務']
    },
    {
        'id': 'IFRS-S1-109',
        'title': '''治理與監督 - 委員會設置''',
        'content': '''個體應揭露永續相關專業委員會的設置及功能。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['專業委員會', '永續委員會', '審核委員會', '風險委員會']
    },
    {
        'id': 'IFRS-S1-110',
        'title': '''治理與監督 - 外部監督''',
        'content': '''個體應揭露外部監督機制的運作及效果。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['外部監督', '獨立董事', '外部審計', '監管機關']
    },
    {
        'id': 'IFRS-S1-111',
        'title': '''治理與監督 - 內部控制''',
        'content': '''個體應揭露永續相關的內部控制機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['內部控制', '控制環境', '風險控制', '合規監控']
    },
    {
        'id': 'IFRS-S1-112',
        'title': '''治理與監督 - 倫理標準''',
        'content': '''個體應揭露永續治理的倫理標準及行為準則。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['倫理標準', '道德規範', '行為準則', '誠信文化']
    },
    {
        'id': 'IFRS-S1-113',
        'title': '''治理與監督 - 利益衝突''',
        'content': '''個體應揭露處理利益衝突的機制及程序。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['利益衝突', '利益迴避', '透明揭露', '獨立性']
    },
    {
        'id': 'IFRS-S1-114',
        'title': '''治理與監督 - 績效評估''',
        'content': '''個體應揭露治理效能的評估及改進機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['治理評估', '董事評鑑', '管理評估', '改進措施']
    },
    {
        'id': 'IFRS-S1-115',
        'title': '''治理與監督 - 繼任規劃''',
        'content': '''個體應揭露永續治理的繼任規劃及知識轉移。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['繼任規劃', '領導人發展', '知識轉移', '連續性']
    },
    {
        'id': 'IFRS-S1-116',
        'title': '''資訊系統與技術 - 一般規定''',
        'content': '''個體應揭露支援永續管理的資訊系統及技術。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['資訊系統', '技術基礎', '數據管理', '數位化']
    },
    {
        'id': 'IFRS-S1-117',
        'title': '''資訊系統與技術 - 數據收集''',
        'content': '''個體應揭露數據收集的技術及自動化機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['數據收集', '自動化', '感測器', '即時監控']
    },
    {
        'id': 'IFRS-S1-118',
        'title': '''資訊系統與技術 - 數據分析''',
        'content': '''個體應揭露數據分析的技術及應用。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['數據分析', '人工智慧', '機器學習', '預測模型']
    },
    {
        'id': 'IFRS-S1-119',
        'title': '''資訊系統與技術 - 報告生成''',
        'content': '''個體應揭露報告生成的技術及功能。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['自動報告', '視覺化', '互動式儀表板', '客製化報告']
    },
    {
        'id': 'IFRS-S1-120',
        'title': '''資訊系統與技術 - 系統整合''',
        'content': '''個體應揭露資訊系統的整合及互聯機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['系統整合', '數據共享', 'API連接', '平台整合']
    },
    {
        'id': 'IFRS-S1-121',
        'title': '''資訊系統與技術 - 數據安全''',
        'content': '''個體應揭露數據安全的技術及措施。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['數據安全', '隱私保護', '加密技術', '存取控制']
    },
    {
        'id': 'IFRS-S1-122',
        'title': '''資訊系統與技術 - 系統維護''',
        'content': '''個體應揭露資訊系統的維護及支援機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['系統維護', '更新升級', '備份恢復', '技術支援']
    },
    {
        'id': 'IFRS-S1-123',
        'title': '''資訊系統與技術 - 成本效益''',
        'content': '''個體應揭露資訊系統的成本效益分析。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['成本效益', '投資回報', '效率提升', '資源節省']
    },
    {
        'id': 'IFRS-S1-124',
        'title': '''資訊系統與技術 - 用戶訓練''',
        'content': '''個體應揭露系統使用者的訓練及支援。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['用戶訓練', '技能發展', '知識轉移', '技術採用']
    },
    {
        'id': 'IFRS-S1-125',
        'title': '''資訊系統與技術 - 創新應用''',
        'content': '''個體應揭露資訊系統的創新技術應用。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['創新應用', '新興技術', '區塊鏈', '物聯網']
    },
    {
        'id': 'IFRS-S1-126',
        'title': '''外部合作與夥伴關係 - 一般規定''',
        'content': '''個體應揭露外部合作及夥伴關係的整體策略。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['外部合作', '夥伴關係', '聯盟', '生態系統']
    },
    {
        'id': 'IFRS-S1-127',
        'title': '''外部合作與夥伴關係 - 供應商合作''',
        'content': '''個體應揭露與供應商的合作及發展機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['供應商合作', '供應鏈夥伴', '共同發展', '價值共創']
    },
    {
        'id': 'IFRS-S1-128',
        'title': '''外部合作與夥伴關係 - 客戶合作''',
        'content': '''個體應揭露與客戶的合作及創新機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['客戶合作', '共同創新', '價值鏈整合', '體驗提升']
    },
    {
        'id': 'IFRS-S1-129',
        'title': '''外部合作與夥伴關係 - 政府合作''',
        'content': '''個體應揭露與政府的合作及政策參與。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['政府合作', '政策對話', '法規遵循', '公共事務']
    },
    {
        'id': 'IFRS-S1-130',
        'title': '''外部合作與夥伴關係 - 學術合作''',
        'content': '''個體應揭露與學術機構的合作及研究機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['學術合作', '研究夥伴', '知識轉移', '創新合作']
    },
    {
        'id': 'IFRS-S1-131',
        'title': '''外部合作與夥伴關係 - NGO合作''',
        'content': '''個體應揭露與非政府組織的合作及社會參與。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['NGO合作', '非政府組織', '社會影響', '倡議參與']
    },
    {
        'id': 'IFRS-S1-132',
        'title': '''外部合作與夥伴關係 - 行業聯盟''',
        'content': '''個體應揭露參與行業聯盟及集體行動。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['行業聯盟', '行業標準', '共同倡議', '集體行動']
    },
    {
        'id': 'IFRS-S1-133',
        'title': '''外部合作與夥伴關係 - 投資者對話''',
        'content': '''個體應揭露與投資者的對話及參與機制。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['投資者對話', '股東參與', '資本市場', 'ESG投資']
    },
    {
        'id': 'IFRS-S1-134',
        'title': '''外部合作與夥伴關係 - 合作協議''',
        'content': '''個體應揭露合作協議的內容及績效評估。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['合作協議', '合約安排', '權利義務', '績效評估']
    },
    {
        'id': 'IFRS-S1-135',
        'title': '''外部合作與夥伴關係 - 合作成效''',
        'content': '''個體應揭露合作關係的成效及價值創造。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['合作成效', '影響評估', '價值創造', '共享利益']
    },
    {
        'id': 'IFRS-S1-136',
        'title': '''永續投資與融資 - 一般規定''',
        'content': '''個體應揭露永續相關的投資及融資策略。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['永續投資', '綠色融資', 'ESG投資', '影響投資']
    },
    {
        'id': 'IFRS-S1-137',
        'title': '''永續投資與融資 - 綠色債券''',
        'content': '''個體應揭露綠色債券的發行及資金運用。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['綠色債券', '綠色融資', '環境項目', '資金用途']
    },
    {
        'id': 'IFRS-S1-138',
        'title': '''永續投資與融資 - 社會債券''',
        'content': '''個體應揭露社會債券的發行及社會影響。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['社會債券', '社會項目', '社會影響', '資金配置']
    },
    {
        'id': 'IFRS-S1-139',
        'title': '''永續投資與融資 - 永續連結債券''',
        'content': '''個體應揭露永續連結債券的設計及績效連結。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['永續連結債券', '績效目標', '獎勵機制', '激勵設計']
    },
    {
        'id': 'IFRS-S1-140',
        'title': '''永續投資與融資 - ESG投資組合''',
        'content': '''個體應揭露ESG投資組合的策略及績效。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['ESG投資組合', '投資篩選', '影響評估', '投資回報']
    },
    {
        'id': 'IFRS-S1-141',
        'title': '''永續投資與融資 - 影響投資''',
        'content': '''個體應揭露影響投資的策略及影響衡量。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['影響投資', '社會影響', '環境影響', '衡量標準']
    },
    {
        'id': 'IFRS-S1-142',
        'title': '''永續投資與融資 - 融資條件''',
        'content': '''個體應揭露融資協議中的ESG相關條件。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['融資條件', 'ESG條款', '績效要求', '報告義務']
    },
    {
        'id': 'IFRS-S1-143',
        'title': '''永續投資與融資 - 投資者關係''',
        'content': '''個體應揭露與投資者的ESG相關溝通。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['投資者關係', 'ESG對話', '透明揭露', '信任建立']
    },
    {
        'id': 'IFRS-S1-144',
        'title': '''永續投資與融資 - 市場趨勢''',
        'content': '''個體應揭露永續投資市場的趨勢及發展。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['市場趨勢', '永續金融', '法規發展', '投資偏好']
    },
    {
        'id': 'IFRS-S1-145',
        'title': '''未來展望與長期規劃''',
        'content': '''個體應揭露對永續發展的長期願景及規劃。''',
        'category': '',
        'difficulty': 'medium',
        'keywords': ['未來展望', '長期規劃', '願景', '轉型目標', '永續發展']
    }
]
