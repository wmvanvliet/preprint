
write_codes = true;
active_buttons = 2;
begin;

picture {
   default_code = "fixation";
   bitmap {
        filename = "fixation_cross.png";
   };
   x = 0; y = 0;
} fixation;

TEMPLATE "stimulus.tem" {
word file code;
"xpfpj" "consonants_xpfpj.png" 20;
"ds^^o" "symbols_ds^^o.png" 30;
"shiia" "word_shiia.png" 10;
"pamppu" "word_pamppu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ q _" "word_pamppu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tenä" "word_tenä.png" 10;
"siitos" "word_siitos.png" 10;
"vv^^" "symbols_vv^^.png" 30;
"rjfcdx" "consonants_rjfcdx.png" 20;
"tcfwdr" "consonants_tcfwdr.png" 20;
"pesula" "word_pesula.png" 10;
"so^^v" "symbols_so^^v.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ^ _" "symbols_so^^v_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"fbtsr" "consonants_fbtsr.png" 20;
"krvvpr" "consonants_krvvpr.png" 20;
"vovdsv" "symbols_vovdsv.png" 30;
"otsoni" "word_otsoni.png" 10;
"jäte" "word_jäte.png" 10;
"vvo^" "symbols_vvo^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ^" "symbols_vvo^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"grgdd" "consonants_grgdd.png" 20;
"d^v^^" "symbols_d^v^^.png" 30;
"kopio" "word_kopio.png" 10;
"voov^v" "symbols_voov^v.png" 30;
"almu" "word_almu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"a _ _ _" "word_almu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rämä" "word_rämä.png" 10;
"dssddd" "symbols_dssddd.png" 30;
"itää" "word_itää.png" 10;
"hihna" "word_hihna.png" 10;
"osinko" "word_osinko.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ y _ _" "word_osinko_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vs^vds" "symbols_vs^vds.png" 30;
"karies" "word_karies.png" 10;
"kopina" "word_kopina.png" 10;
"sarake" "word_sarake.png" 10;
"bzhkd" "consonants_bzhkd.png" 20;
"jdfjs" "consonants_jdfjs.png" 20;
"hautua" "word_hautua.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ u _ _ _" "word_hautua_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"arpoa" "word_arpoa.png" 10;
"vvvd" "symbols_vvvd.png" 30;
"rfclk" "consonants_rfclk.png" 20;
"oosvsd" "symbols_oosvsd.png" 30;
"vwrptk" "consonants_vwrptk.png" 20;
"s^vds^" "symbols_s^vds^.png" 30;
"lohi" "word_lohi.png" 10;
"kaapia" "word_kaapia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"k _ _ _ _ _" "word_kaapia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"noppa" "word_noppa.png" 10;
"qvkjt" "consonants_qvkjt.png" 20;
"riimu" "word_riimu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _ _" "word_riimu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jaardi" "word_jaardi.png" 10;
"syaani" "word_syaani.png" 10;
"spmb" "consonants_spmb.png" 20;
"torium" "word_torium.png" 10;
"bhsx" "consonants_bhsx.png" 20;
"mäntä" "word_mäntä.png" 10;
"kphwjz" "consonants_kphwjz.png" 20;
"gnbtn" "consonants_gnbtn.png" 20;
"säiky" "word_säiky.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _ _" "word_säiky_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dv^oo" "symbols_dv^oo.png" 30;
"pokeri" "word_pokeri.png" 10;
"jymy" "word_jymy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ y" "word_jymy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vahaus" "word_vahaus.png" 10;
"vovso" "symbols_vovso.png" 30;
"estyä" "word_estyä.png" 10;
"särkyä" "word_särkyä.png" 10;
"^vvs^s" "symbols_^vvs^s.png" 30;
"voo^o" "symbols_voo^o.png" 30;
"ässä" "word_ässä.png" 10;
"harjus" "word_harjus.png" 10;
"ripsi" "word_ripsi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"r _ _ _ _" "word_ripsi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"suippo" "word_suippo.png" 10;
"ripeä" "word_ripeä.png" 10;
"ttjglk" "consonants_ttjglk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ n _ _" "consonants_ttjglk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tyköä" "word_tyköä.png" 10;
"tcftcg" "consonants_tcftcg.png" 20;
"köli" "word_köli.png" 10;
"känsä" "word_känsä.png" 10;
"häkä" "word_häkä.png" 10;
"huorin" "word_huorin.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"g _ _ _ _ _" "word_huorin_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kussa" "word_kussa.png" 10;
"gthb" "consonants_gthb.png" 20;
"dfszzp" "consonants_dfszzp.png" 20;
"vipu" "word_vipu.png" 10;
"xggtw" "consonants_xggtw.png" 20;
"tuohus" "word_tuohus.png" 10;
"zfjxqk" "consonants_zfjxqk.png" 20;
"rapsi" "word_rapsi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ s _" "word_rapsi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jtltfj" "consonants_jtltfj.png" 20;
"lotja" "word_lotja.png" 10;
"juhta" "word_juhta.png" 10;
"bfkx" "consonants_bfkx.png" 20;
"zkwnr" "consonants_zkwnr.png" 20;
"^d^^d" "symbols_^d^^d.png" 30;
"odvs^" "symbols_odvs^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ v _ _" "symbols_odvs^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"häät" "word_häät.png" 10;
"s^o^^v" "symbols_s^o^^v.png" 30;
"terska" "word_terska.png" 10;
"syöpyä" "word_syöpyä.png" 10;
"motata" "word_motata.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"x _ _ _ _ _" "word_motata_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ääriin" "word_ääriin.png" 10;
"kolhia" "word_kolhia.png" 10;
"sdvsv" "symbols_sdvsv.png" 30;
"ositus" "word_ositus.png" 10;
"miten" "word_miten.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ e _" "word_miten_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"viipyä" "word_viipyä.png" 10;
"^s^s" "symbols_^s^s.png" 30;
"gongi" "word_gongi.png" 10;
"s^od^" "symbols_s^od^.png" 30;
"kihu" "word_kihu.png" 10;
"gsdx" "consonants_gsdx.png" 20;
"^^od" "symbols_^^od.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ o" "symbols_^^od_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"loraus" "word_loraus.png" 10;
"vdod^" "symbols_vdod^.png" 30;
"katodi" "word_katodi.png" 10;
"kuskus" "word_kuskus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"k _ _ _ _ _" "word_kuskus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"blkxwz" "consonants_blkxwz.png" 20;
"gchqcz" "consonants_gchqcz.png" 20;
"wcjrjq" "consonants_wcjrjq.png" 20;
"dovv^" "symbols_dovv^.png" 30;
"druidi" "word_druidi.png" 10;
"myyty" "word_myyty.png" 10;
"vzkt" "consonants_vzkt.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ k _" "consonants_vzkt_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kolvi" "word_kolvi.png" 10;
"sdosd" "symbols_sdosd.png" 30;
"dovdd" "symbols_dovdd.png" 30;
"vdso" "symbols_vdso.png" 30;
"nieriä" "word_nieriä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ i _ _ _ _" "word_nieriä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rulla" "word_rulla.png" 10;
"lgtjb" "consonants_lgtjb.png" 20;
"seimi" "word_seimi.png" 10;
"klaava" "word_klaava.png" 10;
"oinas" "word_oinas.png" 10;
"^osd" "symbols_^osd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "symbols_^osd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tvzdg" "consonants_tvzdg.png" 20;
"sovo" "symbols_sovo.png" 30;
"kohu" "word_kohu.png" 10;
"qwjvhz" "consonants_qwjvhz.png" 20;
"sovssd" "symbols_sovssd.png" 30;
"säle" "word_säle.png" 10;
"sopu" "word_sopu.png" 10;
"dwzsrc" "consonants_dwzsrc.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ l _ _" "consonants_dwzsrc_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lxtgwm" "consonants_lxtgwm.png" 20;
"xxqhnz" "consonants_xxqhnz.png" 20;
"^vovvs" "symbols_^vovvs.png" 30;
"nisä" "word_nisä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ y" "word_nisä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sodvdv" "symbols_sodvdv.png" 30;
"sdvdvd" "symbols_sdvdvd.png" 30;
"hmff" "consonants_hmff.png" 20;
"sysi" "word_sysi.png" 10;
"uuma" "word_uuma.png" 10;
"d^^do" "symbols_d^^do.png" 30;
"äänne" "word_äänne.png" 10;
"jänne" "word_jänne.png" 10;
"qckq" "consonants_qckq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ q" "consonants_qckq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^v^" "symbols_v^v^.png" 30;
"s^od" "symbols_s^od.png" 30;
"diodi" "word_diodi.png" 10;
"qpvbs" "consonants_qpvbs.png" 20;
"d^sd" "symbols_d^sd.png" 30;
"tdgw" "consonants_tdgw.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ w" "consonants_tdgw_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vaje" "word_vaje.png" 10;
"jxfm" "consonants_jxfm.png" 20;
"ssods" "symbols_ssods.png" 30;
"nurja" "word_nurja.png" 10;
"itiö" "word_itiö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ö" "word_itiö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"fuksi" "word_fuksi.png" 10;
"hamaan" "word_hamaan.png" 10;
"dxfxv" "consonants_dxfxv.png" 20;
"xwnh" "consonants_xwnh.png" 20;
"^dso" "symbols_^dso.png" 30;
"isyys" "word_isyys.png" 10;
"rukki" "word_rukki.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ i" "word_rukki_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^oddo" "symbols_^oddo.png" 30;
"ovvds" "symbols_ovvds.png" 30;
"orpo" "word_orpo.png" 10;
"odo^" "symbols_odo^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "symbols_odo^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^so^d" "symbols_^so^d.png" 30;
"fotoni" "word_fotoni.png" 10;
"ähky" "word_ähky.png" 10;
"oo^v" "symbols_oo^v.png" 30;
"czkdbs" "consonants_czkdbs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"c _ _ _ _ _" "consonants_czkdbs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"salaus" "word_salaus.png" 10;
"vssd" "symbols_vssd.png" 30;
"kortti" "word_kortti.png" 10;
"jbntmc" "consonants_jbntmc.png" 20;
"vuoka" "word_vuoka.png" 10;
"dfdzj" "consonants_dfdzj.png" 20;
"d^dvdd" "symbols_d^dvdd.png" 30;
"tczhj" "consonants_tczhj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ z _ _" "consonants_tczhj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"svvssv" "symbols_svvssv.png" 30;
"äänes" "word_äänes.png" 10;
"zztr" "consonants_zztr.png" 20;
"os^ood" "symbols_os^ood.png" 30;
"päkiä" "word_päkiä.png" 10;
"^d^vs" "symbols_^d^vs.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ o _ _" "symbols_^d^vs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"faksi" "word_faksi.png" 10;
"ovo^s" "symbols_ovo^s.png" 30;
"kiuas" "word_kiuas.png" 10;
"vsd^" "symbols_vsd^.png" 30;
"mkfz" "consonants_mkfz.png" 20;
"äimä" "word_äimä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"e _ _ _" "word_äimä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kuje" "word_kuje.png" 10;
"uima" "word_uima.png" 10;
"vtkdtx" "consonants_vtkdtx.png" 20;
"d^oood" "symbols_d^oood.png" 30;
"ssss" "symbols_ssss.png" 30;
"tyvi" "word_tyvi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ d _ _" "word_tyvi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"duuri" "word_duuri.png" 10;
"tyrä" "word_tyrä.png" 10;
"solmio" "word_solmio.png" 10;
"odote" "word_odote.png" 10;
"s^vo" "symbols_s^vo.png" 30;
"kyteä" "word_kyteä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ä" "word_kyteä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jaos" "word_jaos.png" 10;
"s^dv^" "symbols_s^dv^.png" 30;
"lempo" "word_lempo.png" 10;
"someen" "word_someen.png" 10;
"^d^oos" "symbols_^d^oos.png" 30;
"tämä" "word_tämä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ ä _ _" "word_tämä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"häpy" "word_häpy.png" 10;
"räme" "word_räme.png" 10;
"emätin" "word_emätin.png" 10;
"xkfvmb" "consonants_xkfvmb.png" 20;
"ltqrr" "consonants_ltqrr.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ j _" "consonants_ltqrr_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"täällä" "word_täällä.png" 10;
"erkani" "word_erkani.png" 10;
"ampuja" "word_ampuja.png" 10;
"ndsvw" "consonants_ndsvw.png" 20;
"yöpyä" "word_yöpyä.png" 10;
"voss" "symbols_voss.png" 30;
"o^ss" "symbols_o^ss.png" 30;
"rmmrh" "consonants_rmmrh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ m _ _ _" "consonants_rmmrh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"piip" "word_piip.png" 10;
"vovvvv" "symbols_vovvvv.png" 30;
"kustos" "word_kustos.png" 10;
"xgtt" "consonants_xgtt.png" 20;
"kaste" "word_kaste.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _" "word_kaste_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kysta" "word_kysta.png" 10;
"^sodvv" "symbols_^sodvv.png" 30;
"sävy" "word_sävy.png" 10;
"bwdz" "consonants_bwdz.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _" "consonants_bwdz_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^ss^" "symbols_^ss^.png" 30;
"hioa" "word_hioa.png" 10;
"txhs" "consonants_txhs.png" 20;
"näkö" "word_näkö.png" 10;
"nirso" "word_nirso.png" 10;
"dv^od" "symbols_dv^od.png" 30;
"bxhl" "consonants_bxhl.png" 20;
"uuni" "word_uuni.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"n _ _ _" "word_uuni_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vsd^ds" "symbols_vsd^ds.png" 30;
"dvqdj" "consonants_dvqdj.png" 20;
"btmk" "consonants_btmk.png" 20;
"luusto" "word_luusto.png" 10;
"ylkä" "word_ylkä.png" 10;
"korren" "word_korren.png" 10;
"jqdsmj" "consonants_jqdsmj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ n _" "consonants_jqdsmj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"puida" "word_puida.png" 10;
"silsa" "word_silsa.png" 10;
"uivelo" "word_uivelo.png" 10;
"tykö" "word_tykö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ h _" "word_tykö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pesin" "word_pesin.png" 10;
"osuma" "word_osuma.png" 10;
"svvd" "symbols_svvd.png" 30;
"wnnz" "consonants_wnnz.png" 20;
"eliö" "word_eliö.png" 10;
"rrwj" "consonants_rrwj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ w _" "consonants_rrwj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^vd^vd" "symbols_^vd^vd.png" 30;
"vs^oo" "symbols_vs^oo.png" 30;
"sikhi" "word_sikhi.png" 10;
"dvsvos" "symbols_dvsvos.png" 30;
"ztlrlf" "consonants_ztlrlf.png" 20;
"odos" "symbols_odos.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _" "symbols_odos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sqdq" "consonants_sqdq.png" 20;
"nurin" "word_nurin.png" 10;
"ajos" "word_ajos.png" 10;
"pöty" "word_pöty.png" 10;
"uoma" "word_uoma.png" 10;
"^ddv" "symbols_^ddv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v" "symbols_^ddv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jakaja" "word_jakaja.png" 10;
"lcrzjb" "consonants_lcrzjb.png" 20;
"osdod^" "symbols_osdod^.png" 30;
"näppy" "word_näppy.png" 10;
"^^svd" "symbols_^^svd.png" 30;
"^vod" "symbols_^vod.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v" "symbols_^vod_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"oo^vdv" "symbols_oo^vdv.png" 30;
"hxjlgq" "consonants_hxjlgq.png" 20;
"doosv" "symbols_doosv.png" 30;
"kytkin" "word_kytkin.png" 10;
"psalmi" "word_psalmi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"q _ _ _ _ _" "word_psalmi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vo^^" "symbols_vo^^.png" 30;
"älytä" "word_älytä.png" 10;
"räntä" "word_räntä.png" 10;
"uute" "word_uute.png" 10;
"mxnhh" "consonants_mxnhh.png" 20;
"sv^vvs" "symbols_sv^vvs.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ o _" "symbols_sv^vvs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^vsov" "symbols_v^vsov.png" 30;
"nide" "word_nide.png" 10;
"urut" "word_urut.png" 10;
"bjwqk" "consonants_bjwqk.png" 20;
"^svs^" "symbols_^svs^.png" 30;
"hajan" "word_hajan.png" 10;
"ksrwvk" "consonants_ksrwvk.png" 20;
"opaali" "word_opaali.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"o _ _ _ _ _" "word_opaali_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rwfcq" "consonants_rwfcq.png" 20;
"mgttwx" "consonants_mgttwx.png" 20;
"qgjwzq" "consonants_qgjwzq.png" 20;
"salvaa" "word_salvaa.png" 10;
"hormi" "word_hormi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ r _ _" "word_hormi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"osvs^d" "symbols_osvs^d.png" 30;
"s^^d" "symbols_s^^d.png" 30;
"dvsv" "symbols_dvsv.png" 30;
"poru" "word_poru.png" 10;
"akti" "word_akti.png" 10;
"fwgntw" "consonants_fwgntw.png" 20;
"pyörö" "word_pyörö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ t _ _" "word_pyörö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"iäti" "word_iäti.png" 10;
"pitäen" "word_pitäen.png" 10;
"vvosod" "symbols_vvosod.png" 30;
"zsgj" "consonants_zsgj.png" 20;
"ilkiö" "word_ilkiö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ö" "word_ilkiö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ovdvd" "symbols_ovdvd.png" 30;
"dsdoo^" "symbols_dsdoo^.png" 30;
"rrggj" "consonants_rrggj.png" 20;
"vdvv" "symbols_vdvv.png" 30;
"whpw" "consonants_whpw.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ p _" "consonants_whpw_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kerubi" "word_kerubi.png" 10;
"zpcqc" "consonants_zpcqc.png" 20;
"lblxm" "consonants_lblxm.png" 20;
"säie" "word_säie.png" 10;
"menijä" "word_menijä.png" 10;
"gljt" "consonants_gljt.png" 20;
"pidot" "word_pidot.png" 10;
"säkä" "word_säkä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"l _ _ _" "word_säkä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ilmi" "word_ilmi.png" 10;
"stoola" "word_stoola.png" 10;
"laukka" "word_laukka.png" 10;
"tlzr" "consonants_tlzr.png" 20;
"lähi" "word_lähi.png" 10;
"gxqtx" "consonants_gxqtx.png" 20;
"ahjo" "word_ahjo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ h _ _" "word_ahjo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uiva" "word_uiva.png" 10;
"lakea" "word_lakea.png" 10;
"do^d" "symbols_do^d.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ s" "symbols_do^d_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"mwgz" "consonants_mwgz.png" 20;
"hjqc" "consonants_hjqc.png" 20;
"uuhi" "word_uuhi.png" 10;
"zdvv" "consonants_zdvv.png" 20;
"dsovsd" "symbols_dsovsd.png" 30;
"s^s^s" "symbols_s^s^s.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _" "symbols_s^s^s_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"suti" "word_suti.png" 10;
"pmxwht" "consonants_pmxwht.png" 20;
"kyhmy" "word_kyhmy.png" 10;
"dlgt" "consonants_dlgt.png" 20;
"möly" "word_möly.png" 10;
"lhqt" "consonants_lhqt.png" 20;
"ratamo" "word_ratamo.png" 10;
"lemu" "word_lemu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ v _" "word_lemu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"silaus" "word_silaus.png" 10;
"^^^os" "symbols_^^^os.png" 30;
"nuha" "word_nuha.png" 10;
"dqkcl" "consonants_dqkcl.png" 20;
"afasia" "word_afasia.png" 10;
"hqzll" "consonants_hqzll.png" 20;
"tkcqd" "consonants_tkcqd.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"t _ _ _ _" "consonants_tkcqd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
};
