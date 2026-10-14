"""Invented text for generate-domains.py: the strings each language needs, and the documents' sections.

Everything here is made up for the corpus: no real people, companies or contracts.
"""

# Strings every language needs. Order of the tuple fields is documented where it is read.
PACKS = {
    "en": dict(
        toc="Contents", page="Page", of="of", table="Table", figure="Figure", equation="Equation",
        head=("Item", "Owner", "Due"), rows=[("Draft", "Legal", "3 Feb"), ("Review", "Finance", "10 Feb"), ("Sign-off", "Board", "24 Feb")],
        group=("Phase one", "Phase two"), nested="Details", sig=("Signed for the first party", "Signed for the second party"),
        fn=("Figures are rounded to the nearest unit.", "Defined terms are listed in the glossary."),
        cm=("Reviewer", "Is this wording final?", ("Author", "Yes, unless the board objects.")),
        tr=("Counterparty", "thirty (30) days", "sixty (60) days"), tr2=("Editor", "may", "shall"),
        chart=("Quarterly results", ["Q1", "Q2", "Q3", "Q4"], "Revenue"), math="The relation used throughout is",
        box="Draft: not for circulation", shapes=("Start", "Review", "Approve"),
        form=("Name", "Date", "I agree", ["Option A", "Option B"]), ctrl=("Title", "Click to enter text"), conf="Confidential",
    ),
    "zh": dict(
        toc="目录", page="第", of="页，共", table="表", figure="图", equation="公式",
        head=("事项", "负责人", "期限"), rows=[("起草", "法务部", "2月3日"), ("审核", "财务部", "2月10日"), ("签发", "董事会", "2月24日")],
        group=("第一阶段", "第二阶段"), nested="明细", sig=("甲方（签章）", "乙方（签章）"),
        fn=("数字四舍五入至整数。", "定义的术语见词汇表。"),
        cm=("审核人", "这一表述是否定稿？", ("起草人", "是的，除非董事会提出异议。")),
        tr=("对方律师", "三十日", "六十日"), tr2=("编辑", "可以", "应当"),
        chart=("季度业绩", ["一季度", "二季度", "三季度", "四季度"], "收入"), math="全文使用的关系式为",
        box="草稿，请勿外传", shapes=("开始", "审核", "批准"),
        form=("姓名", "日期", "本人同意", ["选项甲", "选项乙"]), ctrl=("标题", "点击此处输入文字"), conf="保密",
    ),
    "zh-TW": dict(
        toc="目錄", page="第", of="頁，共", table="表", figure="圖", equation="公式",
        head=("事項", "負責人", "期限"), rows=[("起草", "法務組", "2月3日"), ("審核", "會計組", "2月10日"), ("核定", "主任委員", "2月24日")],
        group=("第一階段", "第二階段"), nested="明細", sig=("甲方（簽章）", "乙方（簽章）"),
        fn=("數字四捨五入至整數。", "定義之用語詳見詞彙表。"),
        cm=("審核人", "此文字是否定稿？", ("起草人", "是，除非委員會有異議。")),
        tr=("對造律師", "三十日", "六十日"), tr2=("編輯", "得", "應"),
        chart=("季度成果", ["第一季", "第二季", "第三季", "第四季"], "收入"), math="全文使用之關係式為",
        box="草稿，請勿外傳", shapes=("開始", "審核", "核定"),
        form=("姓名", "日期", "本人同意", ["選項甲", "選項乙"]), ctrl=("標題", "點選此處輸入文字"), conf="機密",
    ),
    "ja": dict(
        toc="目次", page="ページ", of="／", table="表", figure="図", equation="式",
        head=("項目", "担当", "期限"), rows=[("起案", "法務部", "2月3日"), ("審査", "経理部", "2月10日"), ("決裁", "取締役会", "2月24日")],
        group=("第一段階", "第二段階"), nested="明細", sig=("甲（記名押印）", "乙（記名押印）"),
        fn=("数値は整数に丸めている。", "定義した用語は用語集に掲げる。"),
        cm=("確認者", "この文言で確定でしょうか。", ("起案者", "はい、取締役会の異議がなければ確定です。")),
        tr=("相手方弁護士", "三十日", "六十日"), tr2=("編集者", "することができる", "しなければならない"),
        chart=("四半期の業績", ["第1四半期", "第2四半期", "第3四半期", "第4四半期"], "売上高"), math="本書で用いる関係式は次のとおりである",
        box="草案・社外秘", shapes=("開始", "審査", "承認"),
        form=("氏名", "日付", "同意する", ["選択肢A", "選択肢B"]), ctrl=("題名", "ここをクリックして入力"), conf="社外秘",
    ),
    "ko": dict(
        toc="목차", page="쪽", of="/", table="표", figure="그림", equation="수식",
        head=("항목", "담당", "기한"), rows=[("초안", "법무팀", "2월 3일"), ("검토", "재무팀", "2월 10일"), ("승인", "이사회", "2월 24일")],
        group=("1단계", "2단계"), nested="세부 내용", sig=("갑 (서명)", "을 (서명)"),
        fn=("수치는 정수로 반올림하였다.", "정의된 용어는 용어집에 있다."),
        cm=("검토자", "이 표현이 최종입니까?", ("작성자", "예, 이사회가 이의를 제기하지 않으면 최종입니다.")),
        tr=("상대방 변호사", "30일", "60일"), tr2=("편집자", "할 수 있다", "하여야 한다"),
        chart=("분기별 실적", ["1분기", "2분기", "3분기", "4분기"], "매출"), math="이 문서 전체에서 쓰는 관계식은 다음과 같다",
        box="초안, 배포 금지", shapes=("시작", "검토", "승인"),
        form=("성명", "날짜", "동의함", ["선택 가", "선택 나"]), ctrl=("제목", "여기를 눌러 입력"), conf="대외비",
    ),
    "de": dict(
        toc="Inhalt", page="Seite", of="von", table="Tabelle", figure="Abbildung", equation="Gleichung",
        head=("Punkt", "Zuständig", "Frist"), rows=[("Entwurf", "Recht", "3. Feb."), ("Prüfung", "Finanzen", "10. Feb."), ("Freigabe", "Vorstand", "24. Feb.")],
        group=("Phase eins", "Phase zwei"), nested="Einzelheiten", sig=("Für die erste Partei", "Für die zweite Partei"),
        fn=("Die Zahlen sind auf ganze Einheiten gerundet.", "Die definierten Begriffe stehen im Glossar."),
        cm=("Prüfer", "Ist diese Formulierung endgültig?", ("Autor", "Ja, sofern der Vorstand nicht widerspricht.")),
        tr=("Gegenseite", "dreißig (30) Tage", "sechzig (60) Tage"), tr2=("Lektor", "kann", "muss"),
        chart=("Quartalsergebnisse", ["Q1", "Q2", "Q3", "Q4"], "Umsatz"), math="Durchgängig verwendet wird die Beziehung",
        box="Entwurf: nicht weitergeben", shapes=("Start", "Prüfung", "Freigabe"),
        form=("Name", "Datum", "Ich stimme zu", ["Option A", "Option B"]), ctrl=("Titel", "Hier Text eingeben"), conf="Vertraulich",
    ),
    "fr": dict(
        toc="Sommaire", page="Page", of="sur", table="Tableau", figure="Figure", equation="Équation",
        head=("Point", "Responsable", "Échéance"), rows=[("Projet", "Juridique", "3 févr."), ("Relecture", "Finances", "10 févr."), ("Validation", "Conseil", "24 févr.")],
        group=("Première phase", "Seconde phase"), nested="Détails", sig=("Pour la première partie", "Pour la seconde partie"),
        fn=("Les chiffres sont arrondis à l'unité.", "Les termes définis figurent au glossaire."),
        cm=("Relecteur", "Cette rédaction est-elle définitive ?", ("Auteur", "Oui, sauf objection du conseil.")),
        tr=("Conseil adverse", "trente (30) jours", "soixante (60) jours"), tr2=("Éditeur", "peut", "doit"),
        chart=("Résultats trimestriels", ["T1", "T2", "T3", "T4"], "Chiffre d'affaires"), math="La relation utilisée partout est",
        box="Projet : ne pas diffuser", shapes=("Début", "Relecture", "Validation"),
        form=("Nom", "Date", "J'accepte", ["Option A", "Option B"]), ctrl=("Titre", "Cliquez pour saisir du texte"), conf="Confidentiel",
    ),
    "es": dict(
        toc="Índice", page="Página", of="de", table="Tabla", figure="Figura", equation="Ecuación",
        head=("Asunto", "Responsable", "Plazo"), rows=[("Borrador", "Jurídico", "3 feb."), ("Revisión", "Finanzas", "10 feb."), ("Aprobación", "Consejo", "24 feb.")],
        group=("Primera fase", "Segunda fase"), nested="Detalle", sig=("Por la primera parte", "Por la segunda parte"),
        fn=("Las cifras se redondean a la unidad.", "Los términos definidos figuran en el glosario."),
        cm=("Revisor", "¿Es definitiva esta redacción?", ("Autor", "Sí, salvo objeción del consejo.")),
        tr=("Contraparte", "treinta (30) días", "sesenta (60) días"), tr2=("Editor", "puede", "deberá"),
        chart=("Resultados trimestrales", ["T1", "T2", "T3", "T4"], "Ingresos"), math="La relación que se usa en todo el texto es",
        box="Borrador: no difundir", shapes=("Inicio", "Revisión", "Aprobación"),
        form=("Nombre", "Fecha", "Estoy de acuerdo", ["Opción A", "Opción B"]), ctrl=("Título", "Haga clic para escribir"), conf="Confidencial",
    ),
    "ar": dict(
        toc="المحتويات", page="صفحة", of="من", table="جدول", figure="شكل", equation="معادلة",
        head=("البند", "المسؤول", "الموعد"), rows=[("مسودة", "الشؤون القانونية", "٣ فبراير"), ("مراجعة", "المالية", "١٠ فبراير"), ("اعتماد", "المجلس", "٢٤ فبراير")],
        group=("المرحلة الأولى", "المرحلة الثانية"), nested="تفاصيل", sig=("عن الطرف الأول", "عن الطرف الثاني"),
        fn=("الأرقام مقرّبة إلى أقرب وحدة.", "المصطلحات المعرّفة واردة في المسرد."),
        cm=("المراجع", "هل هذه الصياغة نهائية؟", ("الكاتب", "نعم، ما لم يعترض المجلس.")),
        tr=("الطرف المقابل", "ثلاثين (٣٠) يوماً", "ستين (٦٠) يوماً"), tr2=("المحرر", "يجوز", "يجب"),
        chart=("النتائج الفصلية", ["ف١", "ف٢", "ف٣", "ف٤"], "الإيرادات"), math="العلاقة المستخدمة في كامل الوثيقة هي",
        box="مسودة: لا تُوزَّع", shapes=("بداية", "مراجعة", "اعتماد"),
        form=("الاسم", "التاريخ", "أوافق", ["الخيار أ", "الخيار ب"]), ctrl=("العنوان", "انقر لإدخال النص"), conf="سري",
    ),
    "he": dict(
        toc="תוכן עניינים", page="עמוד", of="מתוך", table="טבלה", figure="איור", equation="משוואה",
        head=("סעיף", "אחראי", "מועד"), rows=[("טיוטה", "משפטית", "3 בפברואר"), ("בדיקה", "כספים", "10 בפברואר"), ("אישור", "הדירקטוריון", "24 בפברואר")],
        group=("שלב ראשון", "שלב שני"), nested="פירוט", sig=("בשם הצד הראשון", "בשם הצד השני"),
        fn=("הנתונים מעוגלים ליחידה השלמה הקרובה.", "המונחים המוגדרים מופיעים במילון המונחים."),
        cm=("בודק", "האם זו הנוסחה הסופית?", ("כותב", "כן, אלא אם הדירקטוריון יתנגד.")),
        tr=("הצד שכנגד", "שלושים (30) יום", "שישים (60) יום"), tr2=("עורך", "רשאי", "חייב"),
        chart=("תוצאות רבעוניות", ["רבעון 1", "רבעון 2", "רבעון 3", "רבעון 4"], "הכנסות"), math="הקשר שבו נעשה שימוש לאורך המסמך הוא",
        box="טיוטה: לא להפצה", shapes=("התחלה", "בדיקה", "אישור"),
        form=("שם", "תאריך", "אני מסכים", ["אפשרות א", "אפשרות ב"]), ctrl=("כותרת", "לחצו להזנת טקסט"), conf="חסוי",
    ),
}

# Sections for each document: (heading, [paragraphs]). The key is the document's name.
SECTIONS = {
    "legal-en-court-complaint": ("Complaint for Breach of Contract", "Example County Superior Court, Case No. 00-0000", [
        ("Parties", ["Plaintiff Northwind Example Ltd is a company that supplies office equipment.", "Defendant Contoso Sample Services Ltd agreed to deliver and install that equipment."]),
        ("Facts", ["On 15 March 2025 the parties signed a supply agreement for forty workstations.", "The Defendant delivered twelve workstations and has not delivered the rest."]),
        ("Claim for relief", ["The Plaintiff asks the Court to order delivery of the remaining workstations or damages in their value.", "The Plaintiff also asks for its costs and interest."]),
    ]),
    "legal-fr-contrat": ("Contrat de prestation de services", "Entre les soussignés", [
        ("Objet", ["Le prestataire s'engage à fournir au client des services de maintenance informatique.", "Les services sont décrits à l'annexe 1."]),
        ("Prix et paiement", ["Le client verse une redevance mensuelle fixe, payable à trente jours.", "Tout retard donne lieu à des intérêts au taux légal."]),
        ("Durée et résiliation", ["Le contrat dure douze mois et se renouvelle par tacite reconduction.", "Chaque partie peut y mettre fin avec un préavis de trois mois."]),
    ]),
    "legal-de-kaufvertrag": ("Kaufvertrag über eine Maschine", "Zwischen den nachstehenden Parteien", [
        ("Kaufgegenstand", ["Der Verkäufer verkauft dem Käufer eine gebrauchte Verpackungsmaschine.", "Die Maschine wird in dem Zustand verkauft, in dem sie sich bei der Besichtigung befand."]),
        ("Kaufpreis", ["Der Kaufpreis beträgt 48.000 Euro zuzüglich Umsatzsteuer.", "Der Käufer zahlt den Kaufpreis binnen vierzehn Tagen nach Rechnungsstellung."]),
        ("Gewährleistung", ["Die Gewährleistungsfrist beträgt zwölf Monate ab Übergabe.", "Offene Mängel sind unverzüglich schriftlich anzuzeigen."]),
    ]),
    "legal-zh-complaint": ("民事起诉状", "原告与被告买卖合同纠纷一案", [
        ("当事人", ["原告：示例北方贸易有限公司，住所地在示例市示例区。", "被告：示例南方设备有限公司，住所地在示例市另一区。"]),
        ("诉讼请求", ["请求判令被告交付剩余二十八台设备，或赔偿相应货款损失。", "请求判令被告承担本案诉讼费用。"]),
        ("事实与理由", ["2025年3月15日，双方签订设备买卖合同，约定被告交付四十台设备。", "被告仅交付十二台，经多次催告仍未履行。"]),
    ]),
    "legal-ar-contract": ("عقد توريد معدات", "أُبرم هذا العقد بين الطرفين الآتي بيانهما", [
        ("موضوع العقد", ["يلتزم المورّد بتوريد أربعين جهاز حاسوب إلى المشتري.", "تُحدَّد مواصفات الأجهزة في الملحق الأول."]),
        ("الثمن والدفع", ["يبلغ الثمن الإجمالي مئة وعشرين ألف ريال.", "يدفع المشتري الثمن خلال ثلاثين يوماً من تاريخ الفاتورة."]),
        ("مدة العقد وإنهاؤه", ["مدة العقد سنة واحدة تبدأ من تاريخ التوقيع.", "يجوز لكل طرف إنهاء العقد بإشعار مكتوب قبل ثلاثة أشهر."]),
    ]),
    "finance-en-prospectus": ("Prospectus: Example Growth Fund", "Offer of units to the public", [
        ("The offer", ["The fund offers units at their net asset value plus a small initial charge.", "Investors may buy or sell units on any business day."]),
        ("Risk factors", ["The value of units can fall as well as rise, and investors may get back less than they paid.", "The fund invests in smaller companies, whose prices can move sharply."]),
        ("Charges", ["The annual management charge is 0.75% of the fund's value.", "There is no charge for switching between funds."]),
    ]),
    "finance-de-audit-letter": ("Bestätigungsvermerk des unabhängigen Abschlussprüfers", "An die Gesellschafter der Beispiel Handels GmbH", [
        ("Prüfungsurteil", ["Wir haben den Jahresabschluss der Gesellschaft für das Geschäftsjahr geprüft.", "Nach unserer Beurteilung vermittelt der Jahresabschluss ein den tatsächlichen Verhältnissen entsprechendes Bild."]),
        ("Grundlage", ["Wir haben unsere Prüfung nach den deutschen Grundsätzen ordnungsmäßiger Abschlussprüfung durchgeführt.", "Unsere Verantwortung ist im Abschnitt über die Verantwortung des Abschlussprüfers beschrieben."]),
        ("Verantwortung der Geschäftsführung", ["Die Geschäftsführung ist für die Aufstellung des Jahresabschlusses verantwortlich.", "Sie hat die Fähigkeit zur Fortführung der Unternehmenstätigkeit zu beurteilen."]),
    ]),
    "finance-ja-securities-report": ("有価証券報告書（抜粋）", "架空株式会社 第10期", [
        ("事業の概況", ["当社は事務用機器の販売及び保守を行っている。", "当事業年度の売上高は前期比八パーセント増加した。"]),
        ("経営成績", ["営業利益は前期を上回り、増収増益となった。", "主な要因は保守契約の増加である。"]),
        ("リスク情報", ["為替の変動は仕入原価に影響を及ぼす可能性がある。", "当社は為替予約により影響の軽減に努めている。"]),
    ]),
    "finance-es-informe": ("Informe anual de resultados", "Ejercicio terminado el 31 de diciembre", [
        ("Resumen", ["Los ingresos crecieron un ocho por ciento respecto del ejercicio anterior.", "El margen operativo se mantuvo estable."]),
        ("Resultados por región", ["La región norte aportó el mayor crecimiento.", "Las ventas en el extranjero aumentaron por la entrada de dos nuevos clientes."]),
        ("Perspectivas", ["Para el próximo ejercicio se prevé un crecimiento moderado.", "La dirección mantendrá el control de los costes."]),
    ]),
    "medical-en-clinical-protocol": ("Clinical Study Protocol EX-101", "A randomised, double-blind, placebo-controlled study", [
        ("Study objectives", ["The primary objective is to compare the change in blood pressure between treatment and placebo.", "Secondary objectives include safety and tolerability."]),
        ("Study design", ["Participants are randomised one to one and treated for twelve weeks.", "Visits take place at screening, baseline and weeks 2, 4, 8 and 12."]),
        ("Eligibility", ["Adults aged 18 to 75 with a clinic blood pressure above 140/90 mmHg may take part.", "Pregnant or breastfeeding women are excluded."]),
    ]),
    "medical-en-sop": ("Standard Operating Procedure: Sample Receipt", "SOP-LAB-014, version 3", [
        ("Purpose", ["This procedure describes how the laboratory receives and logs clinical samples.", "It applies to all staff who handle samples."]),
        ("Procedure", ["Check that the label matches the request form before opening the transport box.", "Record the arrival temperature and the time of receipt in the sample log."]),
        ("Deviations", ["Report any sample that arrives warm or damaged to the laboratory manager at once.", "Do not discard a sample until the manager has recorded a decision."]),
    ]),
    "medical-ko-protocol": ("임상시험 계획서 EX-101", "무작위배정, 이중눈가림, 위약대조 시험", [
        ("시험 목적", ["일차 목적은 시험약군과 위약군의 혈압 변화를 비교하는 것이다.", "이차 목적에는 안전성과 내약성이 포함된다."]),
        ("시험 설계", ["시험대상자는 1:1로 무작위 배정되어 12주간 투여받는다.", "방문은 선별, 기저 및 2, 4, 8, 12주에 이루어진다."]),
        ("선정 기준", ["18세 이상 75세 이하이며 진료실 혈압이 140/90 mmHg를 넘는 성인이 참여할 수 있다.", "임신부와 수유부는 제외한다."]),
    ]),
    "medical-zh-TW-sop": ("標準作業程序：檢體接收", "SOP-LAB-014，第3版", [
        ("目的", ["本程序說明實驗室如何接收並登記臨床檢體。", "適用於所有處理檢體之人員。"]),
        ("作業程序", ["開啟運送箱前，應先核對標籤與申請單是否相符。", "於檢體登記簿記錄到達溫度及接收時間。"]),
        ("偏差處理", ["檢體到達時溫度過高或受損者，應立即通報實驗室主管。", "主管記錄處置決定前，不得丟棄檢體。"]),
    ]),
    "academic-en-paper": ("Estimating Delivery Times from Sparse Order Data", "A short communication", [
        ("Introduction", ["Delivery times are hard to predict when few orders have been observed.", "We propose a simple estimator and study its error."]),
        ("Method", ["We model the delay of each order as a random variable with a common mean.", "The estimator shrinks each route's average towards the overall mean."]),
        ("Results", ["On simulated data the estimator reduced the mean squared error by a third.", "On real orders the gain was smaller but consistent."]),
    ]),
    "academic-zh-thesis": ("基于稀疏订单数据的配送时间估计", "硕士学位论文", [
        ("绪论", ["配送时间在订单较少时难以预测，这是物流研究中的常见问题。", "本文提出一种简单的收缩估计方法，并分析其误差。"]),
        ("方法", ["本文将每个订单的延误视为具有共同均值的随机变量。", "估计量将各线路的平均值向总体均值收缩。"]),
        ("结果与讨论", ["在模拟数据上，估计量使均方误差降低了三分之一。", "在真实订单上提升较小，但结果稳定。"]),
    ]),
    "academic-he-paper": ("הערכת זמני אספקה מנתוני הזמנות דלילים", "מאמר קצר", [
        ("מבוא", ["קשה לחזות זמני אספקה כאשר נצפו הזמנות מעטות.", "אנו מציעים אומד פשוט ובוחנים את שגיאתו."]),
        ("שיטה", ["אנו מדמים את העיכוב בכל הזמנה כמשתנה מקרי בעל ממוצע משותף.", "האומד מקרב את ממוצע כל מסלול אל הממוצע הכללי."]),
        ("תוצאות", ["בנתונים מדומים הקטין האומד את שגיאת הריבוע הממוצעת בשליש.", "בהזמנות אמיתיות היה השיפור קטן יותר אך עקבי."]),
    ]),
    "gov-zh-notice": ("关于做好示例年度档案管理工作的通知", "示办发〔2026〕12号", [
        ("一、总体要求", ["各部门要高度重视档案管理工作，明确责任，落实到人。", "要按照统一标准整理、归档和保管各类文件。"]),
        ("二、主要任务", ["完成本部门近三年文件的清点和归档。", "建立档案借阅登记制度，做到来去有据。"]),
        ("三、工作要求", ["请于3月31日前将自查情况报送办公室。", "办公室将适时组织检查。"]),
    ]),
    "gov-fr-reglement": ("Règlement relatif à la gestion des archives", "Article premier à article dix", [
        ("Champ d'application", ["Le présent règlement s'applique à tous les services de l'administration.", "Il couvre les documents papier et numériques."]),
        ("Conservation", ["Les documents sont conservés pendant la durée fixée par le tableau de gestion.", "À l'issue de cette durée, ils sont versés aux archives ou détruits."]),
        ("Communication", ["Tout agent peut consulter les documents nécessaires à ses fonctions.", "La communication à des tiers est soumise à l'autorisation du responsable."]),
    ]),
    "gov-ja-form": ("届出書（様式第1号）", "架空市役所 総務課", [
        ("届出の内容", ["次のとおり届け出ます。", "記載事項に誤りがないことを確認しました。"]),
        ("添付書類", ["本人確認書類の写しを添付してください。", "代理人による届出の場合は委任状も添付してください。"]),
        ("注意事項", ["太枠内を記入してください。", "届出書は窓口または郵送で提出できます。"]),
    ]),
    "gov-zh-TW-announcement": ("公告：預告修正檔案管理辦法草案", "示字第1150000001號", [
        ("主旨", ["預告修正「檔案管理辦法」部分條文草案。", "歡迎各界於期限內提供意見。"]),
        ("依據", ["依行政程序法第一百五十四條第一項規定辦理。", "草案全文公布於本機關網站。"]),
        ("公告事項", ["對本草案內容有任何意見或修正建議者，請於公告之次日起十四日內提出。", "意見請以書面送交本機關法制單位。"]),
    ]),
    "gov-en-form": ("Application for a Parking Permit", "Example City Council, Form PK-1", [
        ("About you", ["Give your name and address exactly as they appear on your vehicle registration.", "Use one form for each vehicle."]),
        ("About your vehicle", ["Tell us the registration number and whether the vehicle is electric.", "Attach a copy of the registration document."]),
        ("Declaration", ["I declare that the information I have given is correct.", "I understand that a false declaration may lead to the permit being withdrawn."]),
    ]),
    "engineering-en-spec": ("Software Requirements Specification: Order Service", "Document ORD-SRS-001, issue 2", [
        ("Scope", ["This specification defines the requirements for the order service.", "It does not cover the payment service."]),
        ("Functional requirements", ["REQ-001: The service shall accept an order only when every line has a quantity above zero.", "REQ-002: The service shall confirm an accepted order within two seconds."]),
        ("Performance requirements", ["REQ-010: The service shall handle one hundred orders per second.", "REQ-011: The service shall lose no accepted order when one node fails."]),
    ]),
    "engineering-de-norm": ("Norm für die Prüfung von Schraubverbindungen", "Ausgabe 2026-01", [
        ("Anwendungsbereich", ["Diese Norm legt Verfahren zur Prüfung von Schraubverbindungen fest.", "Sie gilt nicht für Verbindungen im Druckbehälterbau."]),
        ("Anforderungen", ["Das Anzugsmoment muss mit einem kalibrierten Werkzeug aufgebracht werden.", "Die Prüfung ist nach jeder Wartung zu wiederholen."]),
        ("Prüfbericht", ["Der Prüfbericht muss das Datum, das Werkzeug und das Ergebnis enthalten.", "Abweichungen sind zu begründen."]),
    ]),
    "hr-en-handbook": ("Employee Handbook", "Example Works Ltd", [
        ("Working hours", ["Full-time staff work thirty-seven and a half hours a week.", "Core hours are ten to four, with flexibility around them."]),
        ("Leave", ["Staff receive twenty-five days of paid leave a year, plus public holidays.", "Leave requests go to the line manager at least two weeks ahead."]),
        ("Conduct", ["We expect staff to treat colleagues and customers with respect.", "Concerns can be raised in confidence with the HR team."]),
    ]),
    "hr-ja-contract": ("雇用契約書", "架空株式会社と従業員との間で", [
        ("契約期間", ["雇用期間は期間の定めのないものとする。", "入社日から三か月間を試用期間とする。"]),
        ("勤務条件", ["所定労働時間は一日八時間、週四十時間とする。", "休日は土曜日、日曜日及び国民の祝日とする。"]),
        ("賃金", ["基本給は月額三十万円とし、毎月末日に支払う。", "通勤手当は実費を支給する。"]),
    ]),
    "hr-he-contract": ("הסכם העסקה", "בין החברה לבין העובד", [
        ("תקופת ההעסקה", ["העובד יועסק לתקופה בלתי קצובה.", "שלושת החודשים הראשונים הם תקופת ניסיון."]),
        ("שעות העבודה", ["שעות העבודה הן שמונה שעות ביום וארבעים שעות בשבוע.", "יום המנוחה השבועי הוא יום שבת."]),
        ("שכר", ["השכר החודשי הוא שלושים אלף שקלים חדשים.", "השכר ישולם עד העשירי לכל חודש."]),
    ]),
    "education-en-syllabus": ("Syllabus: Introduction to Data Analysis", "Course DA101, spring term", [
        ("Course aims", ["This course introduces students to the collection and presentation of data.", "No programming experience is needed."]),
        ("Assessment", ["Coursework counts for forty percent of the final mark.", "The remaining sixty percent comes from a written examination."]),
        ("Reading", ["The main text is available from the library as an e-book.", "Further reading is listed at the end of each week's notes."]),
    ]),
    "education-es-lesson-plan": ("Plan de clase: Los ciclos del agua", "Ciencias naturales, quinto grado", [
        ("Objetivos", ["Los alumnos describirán las etapas del ciclo del agua.", "Los alumnos explicarán por qué el agua cambia de estado."]),
        ("Actividades", ["Se inicia con una pregunta sobre dónde va el agua de un charco.", "Después, los alumnos dibujan el ciclo en parejas."]),
        ("Evaluación", ["El profesor revisará los dibujos con una lista de cotejo.", "Cada alumno responderá tres preguntas breves al final de la clase."]),
    ]),
    "education-ar-course": ("وصف المقرر: مدخل إلى تحليل البيانات", "المقرر ت ح ١٠١، الفصل الدراسي الثاني", [
        ("أهداف المقرر", ["يتعرّف الطلاب في هذا المقرر على جمع البيانات وعرضها.", "لا يشترط أي خبرة سابقة في البرمجة."]),
        ("التقييم", ["تشكّل أعمال الفصل أربعين في المئة من الدرجة النهائية.", "وتأتي الستون في المئة الباقية من الاختبار التحريري."]),
        ("المراجع", ["الكتاب الأساسي متاح في المكتبة بصيغة إلكترونية.", "ترد المراجع الإضافية في نهاية مذكرات كل أسبوع."]),
    ]),
    "mixed-en-zh-report": ("Project report 项目报告", "Bilingual summary 双语摘要", [
        ("Overview 概述", ["The project delivered the new order portal on time 新订单门户按时交付。", "Customers can now track deliveries 客户现在可以跟踪配送。"]),
        ("Results 结果", ["Support calls fell by 30% 客服来电下降了百分之三十。", "Order errors fell by half 订单错误减少了一半。"]),
        ("Next steps 后续工作", ["Extend the portal to partners 将门户扩展到合作伙伴。", "Review costs in June 六月复核成本。"]),
    ]),
    "mixed-en-ja-ko-manual": ("Quick start guide / クイックスタート / 빠른 시작", "Model X-200 printer", [
        ("Setup / 設置 / 설치", ["Place the printer on a flat surface 平らな場所に置いてください 평평한 곳에 놓으세요.", "Connect the power cable 電源ケーブルを接続 전원 케이블을 연결하세요."]),
        ("Use / 使い方 / 사용", ["Press the green button to print 緑のボタンで印刷 녹색 버튼을 눌러 인쇄하세요.", "Load A4 paper 用紙をセット 용지를 넣으세요."]),
        ("Help / ヘルプ / 도움말", ["Visit the support page サポートページ 지원 페이지를 방문하세요.", "Keep this guide 本書を保管 안내서를 보관하세요."]),
    ]),
    "mixed-en-ar-contract": ("Services Agreement / اتفاقية خدمات", "Between the parties named below / بين الطرفين المذكورين أدناه", [
        ("Services / الخدمات", ["The Supplier will provide cleaning services / يقدم المورّد خدمات النظافة.", "Services are provided on weekdays / تُقدَّم الخدمات في أيام العمل."]),
        ("Fees / الأتعاب", ["The monthly fee is SAR 12,000 / الأتعاب الشهرية ١٢٬٠٠٠ ريال.", "Invoices are payable in 30 days / تُدفع الفواتير خلال ٣٠ يوماً."]),
        ("Term / المدة", ["The term is one year / مدة الاتفاقية سنة واحدة.", "Either party may end it on notice / يجوز لأي طرف إنهاؤها بإشعار."]),
    ]),
}

# Pools for the long documents' filler: a sentence is one of each, chosen by position.
FILLER = {
    "en": dict(
        subjects=["The Supplier", "The Client", "Each party", "The Contractor", "The Committee", "The Registrar"],
        verbs=["shall keep", "must review", "may request", "will record", "agrees to provide", "is responsible for"],
        objects=["a written record of the work", "the schedule of charges", "access to the premises", "the outstanding invoices", "the agreed specification", "any change notice"],
        tails=["within ten business days", "before the end of each quarter", "at its own cost", "unless the other party objects", "in accordance with the schedule", "to the reasonable satisfaction of the other party"],
    ),
    "zh": dict(
        subjects=["甲方", "乙方", "双方", "项目负责人", "委员会", "审计小组"],
        verbs=["应当保存", "应当审核", "可以要求", "负责记录", "同意提供", "负责落实"],
        objects=["工作的书面记录", "收费明细表", "对场所的进入权限", "未结清的发票", "约定的技术规范", "各项变更通知"],
        tails=["在十个工作日内", "于每季度结束前", "费用自理", "除非对方提出异议", "按照进度安排", "并使对方合理满意"],
    ),
}
