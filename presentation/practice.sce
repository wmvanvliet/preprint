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
"sarake" "word_sarake.png" 10;
"xxfmhj" "consonants_xxfmhj.png" 20;
"tenä" "word_tenä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ e _ _" "word_tenä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tyrä" "word_tyrä.png" 10;
"lhqt" "consonants_lhqt.png" 20;
"tykö" "word_tykö.png" 10;
"dwzsrc" "consonants_dwzsrc.png" 20;
"bkcvxf" "consonants_bkcvxf.png" 20;
"syöpyä" "word_syöpyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _ _" "word_syöpyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvosod" "symbols_vvosod.png" 30;
"whdcmj" "consonants_whdcmj.png" 20;
"ds^^o" "symbols_ds^^o.png" 30;
"piip" "word_piip.png" 10;
"näppy" "word_näppy.png" 10;
"someen" "word_someen.png" 10;
"v^sd" "symbols_v^sd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d" "symbols_v^sd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"äänes" "word_äänes.png" 10;
"viipyä" "word_viipyä.png" 10;
"ositus" "word_ositus.png" 10;
"yöpyä" "word_yöpyä.png" 10;
"kyteä" "word_kyteä.png" 10;
"wlpn" "consonants_wlpn.png" 20;
"mgttwx" "consonants_mgttwx.png" 20;
"afasia" "word_afasia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ z" "word_afasia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"harjus" "word_harjus.png" 10;
"jaardi" "word_jaardi.png" 10;
"^odds" "symbols_^odds.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ o _ _" "symbols_^odds_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dodv" "symbols_dodv.png" 30;
"dssddd" "symbols_dssddd.png" 30;
"tlzr" "consonants_tlzr.png" 20;
"klaava" "word_klaava.png" 10;
"fotoni" "word_fotoni.png" 10;
"qgtqmd" "consonants_qgtqmd.png" 20;
"^o^dd" "symbols_^o^dd.png" 30;
"vsd^" "symbols_vsd^.png" 30;
"brsjjh" "consonants_brsjjh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ x _" "consonants_brsjjh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bfkx" "consonants_bfkx.png" 20;
"ssdvo^" "symbols_ssdvo^.png" 30;
"czkdbs" "consonants_czkdbs.png" 20;
"jakaja" "word_jakaja.png" 10;
"vssov" "symbols_vssov.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ o _" "symbols_vssov_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sävy" "word_sävy.png" 10;
"pxwjv" "consonants_pxwjv.png" 20;
"^osd" "symbols_^osd.png" 30;
"täällä" "word_täällä.png" 10;
"ähky" "word_ähky.png" 10;
"bzhkd" "consonants_bzhkd.png" 20;
"kihu" "word_kihu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "word_kihu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bwdz" "consonants_bwdz.png" 20;
"nnbp" "consonants_nnbp.png" 20;
"nuha" "word_nuha.png" 10;
"ksrwvk" "consonants_ksrwvk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _ _ _" "consonants_ksrwvk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"räme" "word_räme.png" 10;
"lotja" "word_lotja.png" 10;
"dsovsd" "symbols_dsovsd.png" 30;
"^dss" "symbols_^dss.png" 30;
"qwjvhz" "consonants_qwjvhz.png" 20;
"odos" "symbols_odos.png" 30;
"stoola" "word_stoola.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ t _ _ _ _" "word_stoola_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dlgt" "consonants_dlgt.png" 20;
"hxjlgq" "consonants_hxjlgq.png" 20;
"kortti" "word_kortti.png" 10;
"druidi" "word_druidi.png" 10;
"psalmi" "word_psalmi.png" 10;
"tcmggp" "consonants_tcmggp.png" 20;
"säle" "word_säle.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v" "word_säle_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uuma" "word_uuma.png" 10;
"dd^ds^" "symbols_dd^ds^.png" 30;
"ilmi" "word_ilmi.png" 10;
"uoma" "word_uoma.png" 10;
"nisä" "word_nisä.png" 10;
"karies" "word_karies.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ i _ _" "word_karies_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hmvbt" "consonants_hmvbt.png" 20;
"vovdsv" "symbols_vovdsv.png" 30;
"orpo" "word_orpo.png" 10;
"sysi" "word_sysi.png" 10;
"pitäen" "word_pitäen.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ a _ _" "word_pitäen_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dv^oo" "symbols_dv^oo.png" 30;
"särkyä" "word_särkyä.png" 10;
"tkcqd" "consonants_tkcqd.png" 20;
"urut" "word_urut.png" 10;
"svvv" "symbols_svvv.png" 30;
"oss^^" "symbols_oss^^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"o _ _ _ _" "symbols_oss^^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rapsi" "word_rapsi.png" 10;
"erkani" "word_erkani.png" 10;
"lemu" "word_lemu.png" 10;
"do^d" "symbols_do^d.png" 30;
"txhs" "consonants_txhs.png" 20;
"hnxf" "consonants_hnxf.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ f" "consonants_hnxf_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"svvd" "symbols_svvd.png" 30;
"mäntä" "word_mäntä.png" 10;
"oo^^d^" "symbols_oo^^d^.png" 30;
"emätin" "word_emätin.png" 10;
"jxfm" "consonants_jxfm.png" 20;
"vsd^ds" "symbols_vsd^ds.png" 30;
"jymy" "word_jymy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ y" "word_jymy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uuhi" "word_uuhi.png" 10;
"tyköä" "word_tyköä.png" 10;
"kussa" "word_kussa.png" 10;
"ripsi" "word_ripsi.png" 10;
"dqkcl" "consonants_dqkcl.png" 20;
"pamppu" "word_pamppu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ q _" "word_pamppu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"fgrhzh" "consonants_fgrhzh.png" 20;
"kohu" "word_kohu.png" 10;
"sovs" "symbols_sovs.png" 30;
"^s^sso" "symbols_^s^sso.png" 30;
"kopio" "word_kopio.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ o" "word_kopio_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"menijä" "word_menijä.png" 10;
"terska" "word_terska.png" 10;
"ossdd" "symbols_ossdd.png" 30;
"vzkt" "consonants_vzkt.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "consonants_vzkt_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bjwqk" "consonants_bjwqk.png" 20;
"pesin" "word_pesin.png" 10;
"myyty" "word_myyty.png" 10;
"rlfqmh" "consonants_rlfqmh.png" 20;
"voo^o" "symbols_voo^o.png" 30;
"diodi" "word_diodi.png" 10;
"dfszzp" "consonants_dfszzp.png" 20;
"kiuas" "word_kiuas.png" 10;
"hajan" "word_hajan.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ n" "word_hajan_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gongi" "word_gongi.png" 10;
"kaapia" "word_kaapia.png" 10;
"xgtt" "consonants_xgtt.png" 20;
"qmncn" "consonants_qmncn.png" 20;
"qgnqf" "consonants_qgnqf.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ x _ _" "consonants_qgnqf_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^sodvv" "symbols_^sodvv.png" 30;
"hihna" "word_hihna.png" 10;
"mths" "consonants_mths.png" 20;
"rrggj" "consonants_rrggj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ l _ _ _" "consonants_rrggj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"häkä" "word_häkä.png" 10;
"dvqdj" "consonants_dvqdj.png" 20;
"korren" "word_korren.png" 10;
"vsgjfv" "consonants_vsgjfv.png" 20;
"vsds" "symbols_vsds.png" 30;
"v^od" "symbols_v^od.png" 30;
"kaste" "word_kaste.png" 10;
"köli" "word_köli.png" 10;
"silaus" "word_silaus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _ _" "word_silaus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ripeä" "word_ripeä.png" 10;
"osdod^" "symbols_osdod^.png" 30;
"juhta" "word_juhta.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ o _ _" "word_juhta_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tczhj" "consonants_tczhj.png" 20;
"ddsso" "symbols_ddsso.png" 30;
"itiö" "word_itiö.png" 10;
"xggtw" "consonants_xggtw.png" 20;
"nlmlh" "consonants_nlmlh.png" 20;
"sovs^" "symbols_sovs^.png" 30;
"itää" "word_itää.png" 10;
"vtkdtx" "consonants_vtkdtx.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ t _" "consonants_vtkdtx_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"svvssv" "symbols_svvssv.png" 30;
"fxrbbq" "consonants_fxrbbq.png" 20;
"^d^oos" "symbols_^d^oos.png" 30;
"scgxmk" "consonants_scgxmk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ c _ _" "consonants_scgxmk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvv^v" "symbols_vvv^v.png" 30;
"so^^v" "symbols_so^^v.png" 30;
"tyvi" "word_tyvi.png" 10;
"d^v^^" "symbols_d^v^^.png" 30;
"vuoka" "word_vuoka.png" 10;
"tuohus" "word_tuohus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ u _ _" "word_tuohus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pidot" "word_pidot.png" 10;
"doosv" "symbols_doosv.png" 30;
"osdoo^" "symbols_osdoo^.png" 30;
"kopina" "word_kopina.png" 10;
"dvsv" "symbols_dvsv.png" 30;
"nplbr" "consonants_nplbr.png" 20;
"oosvsd" "symbols_oosvsd.png" 30;
"v^vsov" "symbols_v^vsov.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _ _ _" "symbols_v^vsov_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ssds" "symbols_ssds.png" 30;
"^d^^o^" "symbols_^d^^o^.png" 30;
"vosd" "symbols_vosd.png" 30;
"shiia" "word_shiia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"z _ _ _ _" "word_shiia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^oddo" "symbols_^oddo.png" 30;
"do^so" "symbols_do^so.png" 30;
"cnwh" "consonants_cnwh.png" 20;
"ovvs" "symbols_ovvs.png" 30;
"kytkin" "word_kytkin.png" 10;
"ylkä" "word_ylkä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ k _" "word_ylkä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hautua" "word_hautua.png" 10;
"vovvvv" "symbols_vovvvv.png" 30;
"uute" "word_uute.png" 10;
"s^vds^" "symbols_s^vds^.png" 30;
"osvv" "symbols_osvv.png" 30;
"xnbrh" "consonants_xnbrh.png" 20;
"soov" "symbols_soov.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _" "symbols_soov_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^s^d" "symbols_^s^d.png" 30;
"zsgj" "consonants_zsgj.png" 20;
"cmdhv" "consonants_cmdhv.png" 20;
"säkä" "word_säkä.png" 10;
"jänne" "word_jänne.png" 10;
"mplr" "consonants_mplr.png" 20;
"ässä" "word_ässä.png" 10;
"seimi" "word_seimi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ m _" "word_seimi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gmrq" "consonants_gmrq.png" 20;
"lgtjb" "consonants_lgtjb.png" 20;
"suippo" "word_suippo.png" 10;
"xwnh" "consonants_xwnh.png" 20;
"hormi" "word_hormi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ r _ _" "word_hormi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kysta" "word_kysta.png" 10;
"almu" "word_almu.png" 10;
"räntä" "word_räntä.png" 10;
"fuksi" "word_fuksi.png" 10;
"känsä" "word_känsä.png" 10;
"lohi" "word_lohi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ q _" "word_lohi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"säie" "word_säie.png" 10;
"rukki" "word_rukki.png" 10;
"^d^^d" "symbols_^d^^d.png" 30;
"jäte" "word_jäte.png" 10;
"nirso" "word_nirso.png" 10;
"älytä" "word_älytä.png" 10;
"mwgz" "consonants_mwgz.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ g _" "consonants_mwgz_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kuje" "word_kuje.png" 10;
"lwzsh" "consonants_lwzsh.png" 20;
"miten" "word_miten.png" 10;
"häät" "word_häät.png" 10;
"salvaa" "word_salvaa.png" 10;
"syaani" "word_syaani.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _ _ _" "word_syaani_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dssvds" "symbols_dssvds.png" 30;
"osuma" "word_osuma.png" 10;
"eliö" "word_eliö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _" "word_eliö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pzsfrc" "consonants_pzsfrc.png" 20;
"arpoa" "word_arpoa.png" 10;
"wnnz" "consonants_wnnz.png" 20;
"vvvd" "symbols_vvvd.png" 30;
"lcrzjb" "consonants_lcrzjb.png" 20;
"ratamo" "word_ratamo.png" 10;
"opaali" "word_opaali.png" 10;
"zdvv" "consonants_zdvv.png" 20;
"pmrh" "consonants_pmrh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ r _" "consonants_pmrh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"torium" "word_torium.png" 10;
"salaus" "word_salaus.png" 10;
"vahaus" "word_vahaus.png" 10;
"lakea" "word_lakea.png" 10;
"ääriin" "word_ääriin.png" 10;
"otsoni" "word_otsoni.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ü _" "word_otsoni_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^s^s" "symbols_s^s^s.png" 30;
"vsv^" "symbols_vsv^.png" 30;
"vipu" "word_vipu.png" 10;
"ajos" "word_ajos.png" 10;
"sdvd^" "symbols_sdvd^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ^ _ _" "symbols_sdvd^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jaos" "word_jaos.png" 10;
"hjqc" "consonants_hjqc.png" 20;
"xpfpj" "consonants_xpfpj.png" 20;
"häpy" "word_häpy.png" 10;
"pokeri" "word_pokeri.png" 10;
"^d^vs" "symbols_^d^vs.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _" "symbols_^d^vs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jbntmc" "consonants_jbntmc.png" 20;
"duuri" "word_duuri.png" 10;
"kolvi" "word_kolvi.png" 10;
"nurja" "word_nurja.png" 10;
"fwgntw" "consonants_fwgntw.png" 20;
"vs^oo" "symbols_vs^oo.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _ _" "symbols_vs^oo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ztlrlf" "consonants_ztlrlf.png" 20;
"odote" "word_odote.png" 10;
"^o^so" "symbols_^o^so.png" 30;
"d^dvdd" "symbols_d^dvdd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ o _ _" "symbols_d^dvdd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"akti" "word_akti.png" 10;
"isyys" "word_isyys.png" 10;
"kphwjz" "consonants_kphwjz.png" 20;
"uuni" "word_uuni.png" 10;
"qjtnl" "consonants_qjtnl.png" 20;
"blcrm" "consonants_blcrm.png" 20;
"^dvd" "symbols_^dvd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ v _" "symbols_^dvd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rmmrh" "consonants_rmmrh.png" 20;
"luusto" "word_luusto.png" 10;
"kuskus" "word_kuskus.png" 10;
"ampuja" "word_ampuja.png" 10;
"gsdx" "consonants_gsdx.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ x _" "consonants_gsdx_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"iäti" "word_iäti.png" 10;
"^o^o^v" "symbols_^o^o^v.png" 30;
"qpvbs" "consonants_qpvbs.png" 20;
"sodo" "symbols_sodo.png" 30;
"osd^o" "symbols_osd^o.png" 30;
"gxqtx" "consonants_gxqtx.png" 20;
"xmpt" "consonants_xmpt.png" 20;
"v^dv^s" "symbols_v^dv^s.png" 30;
"kustos" "word_kustos.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ü _ _ _" "word_kustos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kolhia" "word_kolhia.png" 10;
"oinas" "word_oinas.png" 10;
"vaje" "word_vaje.png" 10;
"suti" "word_suti.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "word_suti_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"d^oood" "symbols_d^oood.png" 30;
"^s^ovd" "symbols_^s^ovd.png" 30;
"päkiä" "word_päkiä.png" 10;
"mkfz" "consonants_mkfz.png" 20;
"puida" "word_puida.png" 10;
"nqxzt" "consonants_nqxzt.png" 20;
"ssvv" "symbols_ssvv.png" 30;
"qmff" "consonants_qmff.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"q _ _ _" "consonants_qmff_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uivelo" "word_uivelo.png" 10;
"pöty" "word_pöty.png" 10;
"säiky" "word_säiky.png" 10;
"riimu" "word_riimu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _ _" "word_riimu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"faksi" "word_faksi.png" 10;
"^vvsvv" "symbols_^vvsvv.png" 30;
"svsod" "symbols_svsod.png" 30;
"lempo" "word_lempo.png" 10;
"hcpjtf" "consonants_hcpjtf.png" 20;
"^^svd" "symbols_^^svd.png" 30;
"huorin" "word_huorin.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"g _ _ _ _ _" "word_huorin_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"wqchbh" "consonants_wqchbh.png" 20;
"nieriä" "word_nieriä.png" 10;
"rulla" "word_rulla.png" 10;
"pyörö" "word_pyörö.png" 10;
"ahjo" "word_ahjo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ h _ _" "word_ahjo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"noppa" "word_noppa.png" 10;
"estyä" "word_estyä.png" 10;
"siitos" "word_siitos.png" 10;
"poru" "word_poru.png" 10;
"^vd^vd" "symbols_^vd^vd.png" 30;
"hamaan" "word_hamaan.png" 10;
"^o^o^" "symbols_^o^o^.png" 30;
"s^dv^" "symbols_s^dv^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ ^ _ _ _" "symbols_s^dv^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vv^d" "symbols_vv^d.png" 30;
"s^o^^v" "symbols_s^o^^v.png" 30;
"hioa" "word_hioa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ p _" "word_hioa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^^od" "symbols_^^od.png" 30;
"krvvpr" "consonants_krvvpr.png" 20;
"nhlppr" "consonants_nhlppr.png" 20;
"ilkiö" "word_ilkiö.png" 10;
"äänne" "word_äänne.png" 10;
"uima" "word_uima.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"u _ _ _" "word_uima_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"äimä" "word_äimä.png" 10;
"motata" "word_motata.png" 10;
"ndsvw" "consonants_ndsvw.png" 20;
"v^^^^" "symbols_v^^^^.png" 30;
"näkö" "word_näkö.png" 10;
"jtltfj" "consonants_jtltfj.png" 20;
"^dod^" "symbols_^dod^.png" 30;
"^s^^" "symbols_^s^^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ^" "symbols_^s^^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gcfj" "consonants_gcfj.png" 20;
"laukka" "word_laukka.png" 10;
"nvvvz" "consonants_nvvvz.png" 20;
"uiva" "word_uiva.png" 10;
"nurin" "word_nurin.png" 10;
"osvs^d" "symbols_osvs^d.png" 30;
"loraus" "word_loraus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ o _ _ _ _" "word_loraus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pesula" "word_pesula.png" 10;
"nide" "word_nide.png" 10;
"ovdod^" "symbols_ovdod^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ o" "symbols_ovdod^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kyhmy" "word_kyhmy.png" 10;
"rämä" "word_rämä.png" 10;
"sikhi" "word_sikhi.png" 10;
"silsa" "word_silsa.png" 10;
"möly" "word_möly.png" 10;
"katodi" "word_katodi.png" 10;
"s^od" "symbols_s^od.png" 30;
"wqghh" "consonants_wqghh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ h" "consonants_wqghh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lähi" "word_lähi.png" 10;
"^s^s" "symbols_^s^s.png" 30;
"odvs^" "symbols_odvs^.png" 30;
"solmio" "word_solmio.png" 10;
"gnbtn" "consonants_gnbtn.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ t _" "consonants_gnbtn_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hgvgg" "consonants_hgvgg.png" 20;
"sopu" "word_sopu.png" 10;
"tämä" "word_tämä.png" 10;
"svks" "consonants_svks.png" 20;
"osinko" "word_osinko.png" 10;
"kerubi" "word_kerubi.png" 10;
"sclc" "consonants_sclc.png" 20;
"vdov" "symbols_vdov.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _" "symbols_vdov_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
};
